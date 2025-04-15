import datetime
from os import path
import math
import git

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed

from jointformer.model.trainer import JointFormerTrainer
from dataset.static_dataset import StaticTransformDataset
from dataset.vos_dataset import VOSDataset
from dataset.vos_vost_dataset import VOST_Dataset
from dataset.vos_dataset_without_empty import VOSDatasetWithoutEmptyMasks

from util.logger import TensorboardLogger
from util.configuration import Configuration
from util.load_subset import load_sub, load_sub_empty

"""
Initial setup
"""
# Init distributed environment

####
import os

if 'RANK' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '15678'
    print('one device debug.')
else:
    print('DDP training!')
####

distributed.init_process_group(backend="nccl")
print(f'CUDA Device count: {torch.cuda.device_count()}')

# Parse command line arguments
raw_config = Configuration()
raw_config.parse()

if raw_config['benchmark']:
    torch.backends.cudnn.benchmark = True

# Get current git info
repo = git.Repo(".")
git_info = str(repo.active_branch)+' '+str(repo.head.commit.hexsha)

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print(f'I am rank {local_rank} in this world of size {world_size}!')

network_in_memory = None
stages = raw_config['stages']
stages_to_perform = list(stages)
for si, stage in enumerate(stages_to_perform):

    # Set seed to ensure the same initialization
    torch.manual_seed(14159265)
    np.random.seed(14159265)
    random.seed(14159265)

    # Pick stage specific hyperparameters out
    stage_config = raw_config.get_stage_parameters(stage)
    config = dict(**raw_config.args, **stage_config)
    if config['exp_id'] != 'NULL':
        config['exp_id'] = config['exp_id']+'_s%s'%stages[:si+1]

    config['single_object'] = (stage == '0')

    config['num_gpus'] = world_size
    if config['batch_size']//config['num_gpus']*config['num_gpus'] != config['batch_size']:
        raise ValueError('Batch size must be divisible by the number of GPUs.')
    config['batch_size'] //= config['num_gpus']
    config['num_workers'] //= config['num_gpus']
    print(f'We are assuming {config["num_gpus"]} GPUs.')

    config['now_stage'] = stage
    print(f'We are now starting stage {stage}')

    """
    Model related
    """
    if local_rank == 0:
        # Logging
        if config['exp_id'].lower() != 'null':
            print('I will take the role of logging!')
            long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), config['exp_id'])
        else:
            long_id = None
        logger = TensorboardLogger(config['exp_id'], long_id, git_info)
        logger.log_string('hyperpara', str(config))

        # Construct the rank 0 model
        model = JointFormerTrainer(config, logger=logger, 
                        save_path=path.join('saves', long_id, long_id) if long_id is not None else None, 
                        local_rank=local_rank, world_size=world_size).train()
    else:
        # Construct model for other ranks
        model = JointFormerTrainer(config, local_rank=local_rank, world_size=world_size).train()

    # Load pertrained model if needed
    if raw_config['load_checkpoint'] is not None:
        total_iter = model.load_checkpoint(raw_config['load_checkpoint'])
        raw_config['load_checkpoint'] = None
        print('Previously trained model loaded!')
    else:
        total_iter = 0

    if network_in_memory is not None:
        print('I am loading network from the previous stage')
        model.load_network_in_memory(network_in_memory)
        network_in_memory = None
    elif raw_config['load_network'] is not None:
        print('I am loading network from a disk, as listed in configuration')
        model.load_network(raw_config['load_network'])
        raw_config['load_network'] = None

    """
    Dataloader related
    """
    # To re-seed the randomness everytime we start a worker
    def worker_init_fn(worker_id): 
        worker_seed = torch.initial_seed()%(2**31) + worker_id + local_rank*100
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def construct_loader(dataset):
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
        train_loader = DataLoader(dataset, config['batch_size'], sampler=train_sampler, num_workers=config['num_workers'],
                                worker_init_fn=worker_init_fn, drop_last=True)
        return train_sampler, train_loader

    def renew_vos_loader(max_skip, finetune=False):
        # //5 because we only have annotation for every five frames
        yv_dataset = VOSDataset(path.join(yv_root, 'JPEGImages'), 
                            path.join(yv_root, 'Annotations'), max_skip//5, is_bl=False, subset=load_sub('util/subsets/ytb_train.txt'), num_frames=config['num_frames'], finetune=finetune)
        davis_dataset = VOSDataset(path.join(davis_root, 'JPEGImages', '480p'), 
                            path.join(davis_root, 'Annotations', '480p'), max_skip, is_bl=False, subset=load_sub('util/subsets/davis_train.txt'), num_frames=config['num_frames'], finetune=finetune)
        train_dataset = ConcatDataset([davis_dataset]*5 + [yv_dataset])

        print(f'YouTube dataset size: {len(yv_dataset)}')
        print(f'DAVIS dataset size: {len(davis_dataset)}')
        print(f'Concat dataset size: {len(train_dataset)}')
        print(f'Renewed with {max_skip=}')

        return construct_loader(train_dataset)

    def renew_bl_loader(max_skip, finetune=False):
        train_dataset = VOSDataset(path.join(bl_root, 'JPEGImages'), 
                            path.join(bl_root, 'Annotations'), max_skip, is_bl=True, num_frames=config['num_frames'], finetune=finetune)

        print(f'Blender dataset size: {len(train_dataset)}')
        print(f'Renewed with {max_skip=}')

        return construct_loader(train_dataset)

    def renew_vost_loader(max_skip, finetune=False):
        # //5 because we only have annotation for every five frames
        vost_dataset = VOST_Dataset(path.join(vost_root, 'JPEGImages'), 
                            path.join(vost_root, 'Annotations'), max_skip//5, is_bl=False, subset=load_sub('util/subsets/vost_train.txt'), num_frames=config['num_frames'], finetune=finetune,
                            ignore_thresh=config['ignore_thresh'])

        print(f'VOST dataset size: {len(vost_dataset)}')
        print(f'Renewed with {max_skip=}')

        return construct_loader(vost_dataset)

    def renew_mose_loader(max_skip, finetune=False):
        mose_dataset = VOSDataset(path.join(mose_root, 'JPEGImages'), 
                            path.join(mose_root, 'Annotations'), max_skip, is_bl=False, subset=None, num_frames=config['num_frames'], finetune=finetune)

        print(f'MOSE dataset size: {len(mose_dataset)}')
        print(f'Renewed with {max_skip=}')

        return construct_loader(mose_dataset)

    def renew_lvos_loader(max_skip, finetune=False):
        # //5 because we only have annotation for every five frames
        lvos_dataset = VOSDataset(path.join(lvos_root, 'JPEGImages'), 
                            path.join(lvos_root, 'Annotations'), max_skip//5, is_bl=False, subset=None, num_frames=config['num_frames'], finetune=finetune)

        print(f'LVOS dataset size: {len(lvos_dataset)}')
        print(f'Renewed with {max_skip=}')

        return construct_loader(lvos_dataset)

    def renew_visor_loader(max_skip, finetune=False):
        visor_dataset = VOSDataset(path.join(visor_root, 'JPEGImages', '480p'), 
                            path.join(visor_root, 'Annotations', '480p'), max_skip, is_bl=False, subset=load_sub('util/subsets/visor_train.txt'), num_frames=config['num_frames'], finetune=finetune)

        print(f'VISOR dataset size: {len(visor_dataset)}')
        return construct_loader(visor_dataset)

    def renew_burst_loader(max_skip, finetune=False):
        burst_dataset = VOSDataset(path.join(burst_root, 'train', 'JPEGImages'),
                            path.join(burst_root, 'train', 'Annotations'), max_skip, is_bl=False, subset=None, num_frames=config['num_frames'], finetune=finetune)

        print(f'BURST dataset size: {len(burst_dataset)}')
        print(f'Renewed with {max_skip=}')

        return construct_loader(burst_dataset)

    def mega_loader(max_skip, finetune=False):
        davis_dataset = VOSDatasetWithoutEmptyMasks(path.join(davis_root, 'JPEGImages', '480p'), 
                            path.join(davis_root, 'Annotations', '480p'), max_skip, is_bl=False, subset=load_sub('util/subsets/davis_train.txt'), num_frames=config['num_frames'], finetune=finetune,
                            empty_masks=load_sub_empty('util/subsets/davis_empty_masks.txt'))
        yv_dataset = VOSDatasetWithoutEmptyMasks(path.join(yv_root, 'JPEGImages'), 
                            path.join(yv_root, 'Annotations'), max_skip//5, is_bl=False, subset=load_sub('util/subsets/ytb_train.txt'), num_frames=config['num_frames'], finetune=finetune,
                            empty_masks=load_sub_empty('util/subsets/yv_empty_masks.txt'))
        mose_dataset = VOSDatasetWithoutEmptyMasks(path.join(mose_root, 'JPEGImages'), 
                            path.join(mose_root, 'Annotations'), max_skip, is_bl=False, subset=None, num_frames=config['num_frames'], finetune=finetune,
                            empty_masks=load_sub_empty('util/subsets/mose_empty_masks.txt'))
        burst_dataset = VOSDatasetWithoutEmptyMasks(path.join(burst_root, 'train', 'JPEGImages'),
                            path.join(burst_root, 'train', 'Annotations'), max_skip, is_bl=False, subset=None, num_frames=config['num_frames'], finetune=finetune,
                            empty_masks=load_sub_empty('util/subsets/burst_empty_masks.txt'))
        ovis_dataset = VOSDatasetWithoutEmptyMasks(path.join(ovis_root, 'JPEGImages'), 
                            path.join(ovis_root, 'Annotations'), max_skip//3, is_bl=False, subset=None, num_frames=config['num_frames'], finetune=finetune,
                            empty_masks=load_sub_empty('util/subsets/ovis_empty_masks.txt'))
        print(f'DAVIS dataset size: {len(davis_dataset)}')
        print(f'YouTube dataset size: {len(yv_dataset)}')
        print(f'MOSE dataset size: {len(mose_dataset)}')
        print(f'BURST dataset size: {len(burst_dataset)}')
        print(f'OVIS dataset size: {len(ovis_dataset)}')

        train_dataset = ConcatDataset([davis_dataset]*2 + [yv_dataset]*1 + [mose_dataset]*1 + [burst_dataset]*1 + [ovis_dataset]*1)
        print(f'Concat dataset size: {len(train_dataset)}')
        print(f'Renewed with {max_skip=}')

        return construct_loader(train_dataset)

    """
    Dataset related
    """

    """
    These define the training schedule of the distance between frames
    We will switch to max_skip_values[i] once we pass the percentage specified by increase_skip_fraction[i]
    Not effective for stage 0 training
    The initial value is not listed here but in renew_vos_loader(X)
    """
    max_skip_values = [10, 15, 5, 5]

    if stage == '0':    # Static
        static_root = path.expanduser(config['static_root'])
        # format: path, method (style of storing images), mutliplier
        train_dataset = StaticTransformDataset(
            [
                (path.join(static_root, 'fss'), 0, 1),
                (path.join(static_root, 'DUTS-TR'), 1, 1),
                (path.join(static_root, 'DUTS-TE'), 1, 1),
                (path.join(static_root, 'ecssd'), 1, 1),
                (path.join(static_root, 'BIG_small'), 1, 5),
                (path.join(static_root, 'HRSOD_small'), 1, 5),
            ], num_frames=config['num_frames'])
        train_sampler, train_loader = construct_loader(train_dataset)

        print(f'Static dataset size: {len(train_dataset)}')
    elif stage == '1':  # BL30K
        increase_skip_fraction = [0.1, 0.3, 0.8, 100]
        bl_root = path.join(path.expanduser(config['bl_root']))

        train_sampler, train_loader = renew_bl_loader(5)
        renew_loader = renew_bl_loader
    elif stage == '2' or stage == '3':  # YouTube-VOS & DAVIS
        increase_skip_fraction = [0.1, 0.3, 0.9, 100]
        # VOS dataset, 480p is used for both datasets
        yv_root = path.join(path.expanduser(config['yv_root']), 'train_480p')
        davis_root = path.join(path.expanduser(config['davis_root']), '2017', 'trainval')

        train_sampler, train_loader = renew_vos_loader(5)
        renew_loader = renew_vos_loader
    elif stage == '4':  # VOST
        increase_skip_fraction = [0.1, 0.3, 0.9, 100]
        vost_root = path.join(path.expanduser(config['vost_root']))

        train_sampler, train_loader = renew_vost_loader(5)
        renew_loader = renew_vost_loader
    elif stage == '5':  # MOSE
        increase_skip_fraction = [0.1, 0.3, 0.9, 100]
        mose_root = path.join(path.expanduser(config['mose_root']), 'train_480p')

        train_sampler, train_loader = renew_mose_loader(5)
        renew_loader = renew_mose_loader
    elif stage == '6':  # LVOS
        increase_skip_fraction = [0.1, 0.3, 0.9, 100]
        lvos_root = path.join(path.expanduser(config['lvos_root']), 'train_480p')

        train_sampler, train_loader = renew_lvos_loader(5)
        renew_loader = renew_lvos_loader
    elif stage == '7':  # VISOR
        increase_skip_fraction = [0.1, 0.3, 0.9, 100]
        visor_root = path.join(path.expanduser(config['visor_root']))

        train_sampler, train_loader = renew_visor_loader(5)
        renew_loader = renew_visor_loader
    elif stage == '8':  # BURST
        increase_skip_fraction = [0.1, 0.3, 0.9, 100]
        burst_root = path.join(path.expanduser(config['burst_root']))

        train_sampler, train_loader = renew_burst_loader(5)
        renew_loader = renew_burst_loader
    elif stage == '9':  # Mega, Like Cutie
        increase_skip_fraction = [0.1, 0.3, 0.9, 100]
        davis_root = path.join(path.expanduser(config['davis_root']), '2017', 'trainval')
        yv_root = path.join(path.expanduser(config['yv_root']), 'train_480p')
        mose_root = path.join(path.expanduser(config['mose_root']), 'train_480p')
        burst_root = path.join(path.expanduser(config['burst_root']))
        ovis_root = path.join(path.expanduser(config['ovis_root']))

        train_sampler, train_loader = mega_loader(5)
        renew_loader = mega_loader
    else:
        raise NotImplementedError

    ########  val dataset: DAVIS & LVOS  ##########
    def renew_val_loader(max_skip, finetune=False):
        val_lvos_dataset = VOSDataset(path.join(val_lvos_root, 'JPEGImages'), 
                            path.join(val_lvos_root, 'Annotations'), max_skip//5, is_bl=False, subset=None, num_frames=config['num_frames'], finetune=finetune)
        val_davis_dataset = VOSDataset(path.join(val_davis_root, 'JPEGImages', '480p'),
                            path.join(val_davis_root, 'Annotations', '480p'), max_skip, is_bl=False, subset=load_sub('util/subsets/davis_val.txt'), num_frames=config['num_frames'], finetune=finetune)
        val_dataset = ConcatDataset([val_davis_dataset] + [val_lvos_dataset])

        print(f'LVOS Val dataset size: {len(val_lvos_dataset)}')
        print(f'DAVIS Val dataset size: {len(val_davis_dataset)}')
        print(f'Concat dataset size: {len(val_dataset)}')
        print(f'Renewed with {max_skip=}')

        return construct_loader(val_dataset)

    val_max_skip_values = [10, 15, 5, 5]
    val_increase_skip_fraction = [0.1, 0.3, 0.9, 100]
    val_change_skip_iter = [round(config['iterations'] * f) for f in val_increase_skip_fraction]

    val_lvos_root = path.join(path.expanduser(config['lvos_root']), 'valid')
    val_davis_root = path.join(path.expanduser(config['davis_root']), '2017', 'trainval')

    val_sampler, val_loader = renew_val_loader(max_skip=5)

    ##################

    """
    Determine max epoch
    """
    total_epoch = math.ceil(config['iterations']/len(train_loader))
    current_epoch = total_iter // len(train_loader)
    print(f'We approximately use {total_epoch} epochs.')
    if stage != '0':
        change_skip_iter = [round(config['iterations']*f) for f in increase_skip_fraction]
        # Skip will only change after an epoch, not in the middle
        print(f'The skip value will change approximately at the following iterations: {change_skip_iter[:-1]}')

    """
    Starts training
    """
    finetuning = False
    # Need this to select random bases in different workers
    np.random.seed(np.random.randint(2**30-1) + local_rank*100)
    try:
        while total_iter < config['iterations'] + config['finetune']:
            
            # Crucial for randomness! 
            train_sampler.set_epoch(current_epoch)
            current_epoch += 1
            print(f'Current epoch: {current_epoch}')

            # Train loop
            model.train()
            for data in train_loader:

                ############
                if stage!='0' and (((total_iter + 1) % config['log_val_dataset']) == 0):
                    if total_iter >= val_change_skip_iter[0]:
                        while total_iter >= val_change_skip_iter[0]:
                            val_cur_skip = val_max_skip_values[0]
                            val_max_skip_values = val_max_skip_values[1:]
                            val_change_skip_iter = val_change_skip_iter[1:]
                        print(f'Val Dataset changing skip to {val_cur_skip=}')
                        val_sampler, val_loader = renew_val_loader(val_cur_skip)
                        # break

                    model.val()
                    for idx, val_data in enumerate(val_loader):
                        model.val_pass(val_data, total_iter, idx == (len(val_loader) - 1))
                    model.train()
                #############

                # Update skip if needed
                if stage!='0' and total_iter >= change_skip_iter[0]:
                    while total_iter >= change_skip_iter[0]:
                        cur_skip = max_skip_values[0]
                        max_skip_values = max_skip_values[1:]
                        change_skip_iter = change_skip_iter[1:]
                    print(f'Changing skip to {cur_skip=}')
                    train_sampler, train_loader = renew_loader(cur_skip)
                    break

                # fine-tune means fewer augmentations to train the sensory memory
                if config['finetune'] > 0 and not finetuning and total_iter >= config['iterations']:
                    train_sampler, train_loader = renew_loader(cur_skip, finetune=True)
                    finetuning = True
                    model.save_network_interval = 1000
                    break

                model.do_pass(data, total_iter)
                total_iter += 1

                if total_iter >= config['iterations'] + config['finetune']:
                    break
    finally:
        if not config['debug'] and model.logger is not None and total_iter>5000:
            model.save_network(total_iter)
            model.save_checkpoint(total_iter)

    network_in_memory = model.JointFormer.module.state_dict()

distributed.destroy_process_group()