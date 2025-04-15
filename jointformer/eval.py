import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from jointformer.inference.data.test_datasets import DAVIS_TestDataset, YouTubeVOS_TestDataset, LongVideo_TestDataset
from jointformer.inference.data.test_datasets import VOST_TestDataset, MOSE_TestDataset, LVOS_TestDataset, VISOR_TestDataset, BURST_TestDataset
from jointformer.inference.data.mask_mapper import MaskMapper
from jointformer.model.network import JointFormer
from jointformer.inference.inference_core import InferenceCore

from progressbar import progressbar

try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')
# os.environ['CUDA_VISIBLE_DEVICES'] = "7"


"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default=None)

# Data options
# For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
parser.add_argument('--generic_path')

parser.add_argument('--data_root', default="../", help='Path to the dataset root')
parser.add_argument('--dataset', default='D17', choices=['D16','D17','Y18','Y19','VOST','MOSE','VISOR','LVOS', 'BURST','G'])
parser.add_argument('--split', default='val', choices=['val', 'test'], help='split of the dataset to eval')
parser.add_argument('--output', default=None)
parser.add_argument('--save_all', action='store_true', 
            help='Save all frames. Useful only in YouTubeVOS/long-time video', )

parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')

# Model options
parser.add_argument('--update_block_depth', default=2, type=int, help='the depth of convmae.block4')
parser.add_argument('--enhance_block_depth', default=1, type=int, help='the depth of convmae.block5')

# Top-K options
parser.add_argument('--topk_num', type=int, default=30, help='TopK number')
parser.add_argument('--topk_range', type=str, default='all_frames', choices=['every_frame', 'all_frames'], help='choose topK range?')
parser.add_argument('--topk_block', type=str, default='all_blocks', choices=['last_block', 'all_blocks', 'None', 'half'],
                    help='do topK in each block or the last block? Set None if not use TopK')
parser.add_argument('--topk_clstoken', type=str, default='N', choices=['Y', 'N'], help='mix-attn for cls token do topK ?')

# memory options
parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
parser.add_argument('--include_last', help='include last frame as temporary memory?', type=bool, default=True)  # add
parser.add_argument('--max_memory_frames', help='max frames save in MemoryBank', type=int, default=5)    # add

# Multi-scale options
parser.add_argument('--save_scores', action='store_true')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--size', default=480, type=int, 
            help='Resize the shorter side to this size. -1 to use original resolution. ')

args = parser.parse_args()
config = vars(args)

if args.output is None:
    args.output = f'./output/{args.dataset}_{args.split}'
    print(f'Output path not provided. Defaulting to {args.output}')
print(f'Results will be save to {args.output}')

"""
Data preparation
"""
data_path_map = {
    "D16": 'DAVIS/2016',
    "D17": 'DAVIS/2017',
    "Y18": 'YouTubeVOS-2018',
    "Y19": 'YouTubeVOS-2019',
    "VOST": 'VOST',
    "MOSE": 'MOSE',
    "VISOR": 'VISOR',
    "LVOS": 'LVOS',
    "BURST": 'BURST',
    "G": args.generic_path,
}
data_path = os.path.join(args.data_root, data_path_map[args.dataset])

save_in_Annotations = args.dataset.startswith('Y1') or args.dataset == "LVOS" or args.dataset == "BURST"

if save_in_Annotations or args.save_scores:
    out_path = path.join(args.output, 'Annotations')
else:
    out_path = args.output

if args.dataset == 'Y18' or args.dataset == 'Y19':
    if args.split == 'val':
        args.split = 'valid'
        meta_dataset = YouTubeVOS_TestDataset(data_root=data_path, split='valid', size=args.size)
    elif args.split == 'test':
        meta_dataset = YouTubeVOS_TestDataset(data_root=data_path, split='test', size=args.size)
    else:
        raise NotImplementedError

elif args.dataset == 'D16' or args.dataset == 'D17':
    if args.dataset == 'D16':
        if args.split == 'val':
            # Set up Dataset, a small hack to use the image set in the 2017 folder because the 2016 one is of a different format
            meta_dataset = DAVIS_TestDataset(data_path, imset='../../2017/trainval/ImageSets/2016/val.txt', size=args.size)
        else:
            raise NotImplementedError
        palette = None
    elif args.dataset == 'D17':
        if args.split == 'val':
            meta_dataset = DAVIS_TestDataset(path.join(data_path, 'trainval'), imset='2017/val.txt', size=args.size)
        elif args.split == 'test':
            meta_dataset = DAVIS_TestDataset(path.join(data_path, 'test-dev'), imset='2017/test-dev.txt', size=args.size)
        else:
            raise NotImplementedError

elif args.dataset == 'VOST':
    if args.split == 'val':
        meta_dataset = VOST_TestDataset(data_path, imset='val.txt', size=args.size)
    else:
        raise NotImplementedError

elif args.dataset == 'MOSE':
    if args.split == 'val':
        args.split = 'valid'
        meta_dataset = MOSE_TestDataset(data_root=data_path, split='valid', size=args.size)
    elif args.split == 'test':
        meta_dataset = MOSE_TestDataset(data_root=data_path, split='test', size=args.size)
    else:
        raise NotImplementedError

elif args.dataset == 'VISOR':
    if args.split == 'val':
        meta_dataset = VISOR_TestDataset(data_root=data_path, imset='2022/val.txt', size=args.size)
    else:
        raise NotImplementedError

elif args.dataset == 'LVOS':
    if args.split == 'val':
        args.split = 'valid'
        meta_dataset = LVOS_TestDataset(data_root=data_path, split='valid', size=args.size) # d83wYdy0
    elif args.split == 'test':
        meta_dataset = LVOS_TestDataset(data_root=data_path, split='test', size=args.size)
    else:
        raise NotImplementedError

elif args.dataset == "BURST":
    if args.split == 'val':
        meta_dataset = BURST_TestDataset(data_root=data_path, split='val', size=args.size)
    elif args.split == 'test':
        meta_dataset = BURST_TestDataset(data_root=data_path, split='test', size=args.size)
    else:
        raise NotImplementedError

elif args.dataset == 'G':
    meta_dataset = LongVideo_TestDataset(data_root=data_path, size=args.size)
    if not args.save_all:
        args.save_all = True
        print('save_all is forced to be true in generic evaluation mode.')
else:
    raise NotImplementedError

torch.autograd.set_grad_enabled(False)

# Set up loader
meta_loader = meta_dataset.get_datasets()

# Load our checkpoint
network = JointFormer(config, args.model).cuda().eval()
if args.model is not None:
    model_weights = torch.load(args.model)
    network.load_weights(model_weights, init_as_zero_if_needed=True)
    print(f'Loading model weight from {args.model} finish.')
else:
    print('No model loaded.')

total_process_time = 0
total_frames = 0

# Start eval
for vid_reader in progressbar(meta_loader, max_value=len(meta_dataset), redirect_stdout=True):

    loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
    vid_name = vid_reader.vid_name
    vid_length = len(loader)

    mapper = MaskMapper()
    processor = InferenceCore(network, config=config)
    first_mask_loaded = False

    for ti, data in enumerate(loader):
        with torch.cuda.amp.autocast(enabled=not args.benchmark):
            rgb = data['rgb'].cuda()[0]
            msk = data.get('mask')
            info = data['info']
            frame = info['frame'][0]
            shape = info['shape']
            need_resize = info['need_resize'][0]

            """
            For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
            Seems to be very similar in testing as my previous timing method 
            with two cuda sync + time.time() in STCN though 
            """
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            if not first_mask_loaded:
                if msk is not None:
                    first_mask_loaded = True
                else:
                    # no point to do anything without a mask
                    continue

            if args.flip:
                rgb = torch.flip(rgb, dims=[-1])
                msk = torch.flip(msk, dims=[-1]) if msk is not None else None

            # Map possibly non-continuous labels to continuous ones
            if msk is not None:
                msk, labels = mapper.convert_mask(msk[0].numpy())
                msk = torch.Tensor(msk).cuda()
                if need_resize:
                    msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                processor.set_all_labels(list(mapper.remappings.values()))
            else:
                labels = None

            # Run the model on this frame
            prob = processor.step(rgb, msk, labels, end=(ti==vid_length-1))

            # Upsample to original size if needed
            if need_resize:
                prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

            end.record()
            torch.cuda.synchronize()
            total_process_time += (start.elapsed_time(end)/1000)
            total_frames += 1

            if args.flip:
                prob = torch.flip(prob, dims=[-1])

            # Probability mask -> index mask
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            if args.save_scores:
                prob = (prob.detach().cpu().numpy()*255).astype(np.uint8)

            # Save the mask
            if args.save_all or info['save'][0]:
                this_out_path = path.join(out_path, vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                out_mask = mapper.remap_index_mask(out_mask)
                out_img = Image.fromarray(out_mask)
                if vid_reader.get_palette() is not None:
                    out_img.putpalette(vid_reader.get_palette())
                out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

            if args.save_scores:
                np_path = path.join(args.output, 'Scores', vid_name)
                os.makedirs(np_path, exist_ok=True)
                if ti==len(loader)-1:
                    hkl.dump(mapper.remappings, path.join(np_path, f'backward.hkl'), mode='w')
                if args.save_all or info['save'][0]:
                    hkl.dump(prob, path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')


print(f'Total processing time: {total_process_time}')
print(f'Total processed frames: {total_frames}')
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')

if not args.save_scores:
    if save_in_Annotations:
        print(f'Making zip for {args.dataset} {args.split}...')
        shutil.make_archive(path.join(args.output, path.basename(args.output)), 'zip', args.output, 'Annotations')
    elif args.dataset == 'D17' and args.split == 'test':
        print('Making zip for DAVIS test-dev...')
        shutil.make_archive(args.output, 'zip', args.output)
        shutil.move(f"{args.output}.zip", args.output)
    elif args.dataset == 'MOSE':
        print('Making zip for MOSE...')
        shutil.make_archive(args.output, 'zip', args.output)
        shutil.move(f"{args.output}.zip", args.output)

import json
with open(os.path.join(args.output, f"{args.output.split('/')[-1]}.json"), mode="w") as f:
    config['FPS'] = total_frames / total_process_time
    config['Max allocated memory (MB)'] = torch.cuda.max_memory_allocated() / (2**20)
    f.write(json.dumps(config, indent=4))
    f.write("\n")
    output_dir = config['output']
    print(f'config saved in {output_dir}.')