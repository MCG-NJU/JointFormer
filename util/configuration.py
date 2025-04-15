from argparse import ArgumentParser


def none_or_default(x, default):
    return x if x is not None else default

class Configuration():
    def parse(self, unknown_arg_ok=False):
        parser = ArgumentParser()

        # ----cls_token, backbone_mixattn_convmae_v1_clsToken_v3----#
        parser.add_argument('--backbone_update_clsToken_prob', default=0.6, type=float, help='if the cls_token, updated with frame_with mask in convmae.block3')
        parser.add_argument('--update_block_depth', default=2, type=int, help='the depth of convmae.block4')
        parser.add_argument('--enhance_block_depth', default=1, type=int, help='the depth of convmae.block5')

        # Enable torch.backends.cudnn.benchmark -- Faster in some cases, test in your own environment
        parser.add_argument('--benchmark', action='store_true')
        parser.add_argument('--no_amp', action='store_true')

        # Data parameters
        parser.add_argument('--static_root', help='Static training data root', default='../static')
        parser.add_argument('--bl_root', help='Blender training data root', default='../BL30K')
        parser.add_argument('--yv_root', help='YouTubeVOS training data root', default='../YouTubeVOS-2019')
        parser.add_argument('--davis_root', help='DAVIS training data root', default='../DAVIS')
        parser.add_argument('--vost_root', help='VOST training data root', default='../VOST')
        parser.add_argument('--mose_root', help='MOSE training data root', default='../MOSE')
        parser.add_argument('--lvos_root', help='LVOS training data root', default='../LVOS')
        parser.add_argument('--visor_root', help='VISOR training data root', default='../VISOR')
        parser.add_argument('--burst_root', help='BURST training data root', default='../BURST')
        parser.add_argument('--ovis_root', help='OVIS training data root', default='../OVIS-VOS-train')
        parser.add_argument('--num_workers', help='Total number of dataloader workers across all GPUs processes', type=int, default=40)

        """
        Stage-specific learning parameters
        Batch sizes are effective -- you don't have to scale them when you scale the number processes
        fine-tune means fewer augmentations to train the sensory memory
        """
        parser.add_argument('--stages', help='Training stage (0-static images, 1-Blender dataset, 2-DAVIS+YouTubeVOS)', default='02')

        # Stage 0, static images (37372)
        parser.add_argument('--s0_batch_size', default=16, type=int)
        parser.add_argument('--s0_iterations', default=150000, type=int)
        parser.add_argument('--s0_finetune', default=0, type=int)
        parser.add_argument('--s0_steps', nargs="*", default=[], type=int)
        parser.add_argument('--s0_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s0_num_ref_frames', default=2, type=int)
        parser.add_argument('--s0_num_frames', default=3, type=int)
        parser.add_argument('--s0_start_warm', default=20000, type=int)
        parser.add_argument('--s0_end_warm', default=70000, type=int)

        # Stage 1, BL30K (29989)
        parser.add_argument('--s1_batch_size', default=8, type=int)
        parser.add_argument('--s1_iterations', default=250000, type=int)
        parser.add_argument('--s1_finetune', default=0, type=int)
        parser.add_argument('--s1_steps', nargs="*", default=[200000], type=int)
        parser.add_argument('--s1_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s1_num_ref_frames', default=3, type=int)
        parser.add_argument('--s1_num_frames', default=8, type=int)
        parser.add_argument('--s1_start_warm', default=20000, type=int)
        parser.add_argument('--s1_end_warm', default=70000, type=int)

        # Stage 2, DAVIS+YoutubeVOS (5*60 + 3472), longer
        parser.add_argument('--s2_batch_size', default=8, type=int)
        parser.add_argument('--s2_iterations', default=150000, type=int)
        parser.add_argument('--s2_finetune', default=10000, type=int)
        parser.add_argument('--s2_steps', nargs="*", default=[120000], type=int)
        parser.add_argument('--s2_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s2_num_ref_frames', default=3, type=int)
        parser.add_argument('--s2_num_frames', default=8, type=int)
        parser.add_argument('--s2_start_warm', default=20000, type=int)
        parser.add_argument('--s2_end_warm', default=70000, type=int)

        # Stage 3, DAVIS+YoutubeVOS (5*60 + 3472), shorter
        parser.add_argument('--s3_batch_size', default=8, type=int)
        parser.add_argument('--s3_iterations', default=100000, type=int)
        parser.add_argument('--s3_finetune', default=10000, type=int)
        parser.add_argument('--s3_steps', nargs="*", default=[80000], type=int)
        parser.add_argument('--s3_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s3_num_ref_frames', default=3, type=int)
        parser.add_argument('--s3_num_frames', default=8, type=int)
        parser.add_argument('--s3_start_warm', default=20000, type=int)
        parser.add_argument('--s3_end_warm', default=70000, type=int)

        # Stage 4, VOST (572)
        parser.add_argument('--s4_batch_size', default=8, type=int)
        parser.add_argument('--s4_iterations', default=20000, type=int)
        parser.add_argument('--s4_finetune', default=5000, type=int)
        parser.add_argument('--s4_steps', nargs="*", default=[14000], type=int)
        parser.add_argument('--s4_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s4_num_ref_frames', default=3, type=int)
        parser.add_argument('--s4_num_frames', default=8, type=int)
        parser.add_argument('--s4_start_warm', default=4000, type=int)
        parser.add_argument('--s4_end_warm', default=13000, type=int)
        parser.add_argument('--ignore_thresh', default=0.2, type=float, help='VOST: Ignore Region/foreground pixels')

        # Stage 5, MOSE (1507)
        parser.add_argument('--s5_batch_size', default=8, type=int)
        parser.add_argument('--s5_iterations', default=150000, type=int)
        parser.add_argument('--s5_finetune', default=10000, type=int)
        parser.add_argument('--s5_steps', nargs="*", default=[120000], type=int)
        parser.add_argument('--s5_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s5_num_ref_frames', default=3, type=int)
        parser.add_argument('--s5_num_frames', default=8, type=int)
        parser.add_argument('--s5_start_warm', default=20000, type=int)
        parser.add_argument('--s5_end_warm', default=70000, type=int)

        # Stage 6, LVOS (120)
        parser.add_argument('--s6_batch_size', default=8, type=int)
        parser.add_argument('--s6_iterations', default=20000, type=int)
        parser.add_argument('--s6_finetune', default=5000, type=int)
        parser.add_argument('--s6_steps', nargs="*", default=[14000], type=int)
        parser.add_argument('--s6_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s6_num_ref_frames', default=3, type=int)
        parser.add_argument('--s6_num_frames', default=8, type=int)
        parser.add_argument('--s6_start_warm', default=4000, type=int)
        parser.add_argument('--s6_end_warm', default=13000, type=int)

        # Stage 7, VISOR (5309)
        parser.add_argument('--s7_batch_size', default=8, type=int)
        parser.add_argument('--s7_iterations', default=150000, type=int)
        parser.add_argument('--s7_finetune', default=10000, type=int)
        parser.add_argument('--s7_steps', nargs="*", default=[120000], type=int)
        parser.add_argument('--s7_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s7_num_ref_frames', default=3, type=int)
        parser.add_argument('--s7_num_frames', default=8, type=int)
        parser.add_argument('--s7_start_warm', default=20000, type=int)
        parser.add_argument('--s7_end_warm', default=70000, type=int)
        
        # Stage 8, BURST (500)
        parser.add_argument('--s8_batch_size', default=8, type=int)
        parser.add_argument('--s8_iterations', default=150000, type=int)
        parser.add_argument('--s8_finetune', default=10000, type=int)
        parser.add_argument('--s8_steps', nargs="*", default=[120000], type=int)
        parser.add_argument('--s8_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s8_num_ref_frames', default=3, type=int)
        parser.add_argument('--s8_num_frames', default=8, type=int)
        parser.add_argument('--s8_start_warm', default=20000, type=int)
        parser.add_argument('--s8_end_warm', default=70000, type=int)

        # Stage 9, Mega: DAVIS & YouTubeVOS & MOSE & BURST & OVIS
        parser.add_argument('--s9_batch_size', default=8, type=int)
        parser.add_argument('--s9_iterations', default=200000, type=int)
        parser.add_argument('--s9_finetune', default=10000, type=int)
        parser.add_argument('--s9_steps', nargs="*", default=[160000, 180000], type=int)
        parser.add_argument('--s9_lr', help='Initial learning rate', default=1e-5, type=float)
        parser.add_argument('--s9_num_ref_frames', default=3, type=int)
        parser.add_argument('--s9_num_frames', default=8, type=int)
        parser.add_argument('--s9_start_warm', default=30000, type=int)
        parser.add_argument('--s9_end_warm', default=10000, type=int)

        # Optimizer
        parser.add_argument('--gamma', help='LR := LR*gamma at every decay step', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.05, type=float)
        parser.add_argument('--optimizer', type=str, default='backbone_scale', choices=['default', 'backbone_scale'])

        # Loading
        parser.add_argument('--load_network', help='Path to pretrained network weight only')
        parser.add_argument('--load_checkpoint', help='Path to the checkpoint file, including network, optimizer and such')

        # Logging information
        parser.add_argument('--log_val_dataset', default=500, type=int)
        parser.add_argument('--log_text_interval', default=100, type=int)
        parser.add_argument('--log_image_interval', default=1000, type=int)
        parser.add_argument('--save_network_interval', default=25000, type=int)
        parser.add_argument('--save_checkpoint_interval', default=50000, type=int)
        parser.add_argument('--exp_id', help='Experiment UNIQUE id, use NULL to disable logging to tensorboard', default='NULL')
        parser.add_argument('--debug', help='Debug mode which logs information more often', action='store_true')

        # # Multiprocessing parameters, not set by users
        # parser.add_argument('--local_rank', default=0, type=int, help='Local rank of this process')

        if unknown_arg_ok:
            args, _ = parser.parse_known_args()
            self.args = vars(args)
        else:
            self.args = vars(parser.parse_args())

        self.args['amp'] = not self.args['no_amp']

        # check if the stages are valid
        stage_to_perform = list(self.args['stages'])
        for s in stage_to_perform:
            if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                raise NotImplementedError

    def get_stage_parameters(self, stage):
        parameters = {
            'batch_size': self.args['s%s_batch_size'%stage],
            'iterations': self.args['s%s_iterations'%stage],
            'finetune': self.args['s%s_finetune'%stage],
            'steps': self.args['s%s_steps'%stage],
            'lr': self.args['s%s_lr'%stage],
            'num_ref_frames': self.args['s%s_num_ref_frames'%stage],
            'num_frames': self.args['s%s_num_frames'%stage],
            'start_warm': self.args['s%s_start_warm'%stage],
            'end_warm': self.args['s%s_end_warm'%stage],
        }

        return parameters

    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, value):
        self.args[key] = value

    def __str__(self):
        return str(self.args)
