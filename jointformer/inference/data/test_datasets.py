import os
from os import path
import json

from jointformer.inference.data.video_reader import VideoReader


class LongVideo_TestDataset:
    def __init__(self, data_root, size=-1):
        self.image_dir = path.join(data_root, 'JPEGImages')
        self.mask_dir = path.join(data_root, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                to_save = [
                    name[:-4] for name in os.listdir(path.join(self.mask_dir, video))
                ],
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)


class DAVIS_TestDataset:
    def __init__(self, data_root, imset='2017/val.txt', size=-1):
        if size != 480:
            self.image_dir = path.join(data_root, 'JPEGImages', 'Full-Resolution')
            self.mask_dir = path.join(data_root, 'Annotations', 'Full-Resolution')
            if not path.exists(self.image_dir):
                print(f'{self.image_dir} not found. Look at other options.')
                self.image_dir = path.join(data_root, 'JPEGImages', '1080p')
                self.mask_dir = path.join(data_root, 'Annotations', '1080p')
            assert path.exists(self.image_dir), 'path not found'
        else:
            self.image_dir = path.join(data_root, 'JPEGImages', '480p')
            self.mask_dir = path.join(data_root, 'Annotations', '480p')
        self.size_dir = path.join(data_root, 'JPEGImages', '480p')
        self.size = size

        with open(path.join(data_root, 'ImageSets', imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                size_dir=path.join(self.size_dir, video),
            )

    def __len__(self):
        return len(self.vid_list)


class YouTubeVOS_TestDataset:
    def __init__(self, data_root, split, size=480):
        self.image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        if not os.path.exists(self.image_dir):
            print('All frame not exists.')
            self.image_dir = path.join(data_root, split, 'JPEGImages')
            assert os.path.exists(self.image_dir)
        self.mask_dir = path.join(data_root, split, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))
        self.req_frame_list = {}

        with open(path.join(data_root, split, 'meta.json')) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)['videos']

            for vid in self.vid_list:
                req_frames = []
                objects = meta[vid]['objects']
                for value in objects.values():
                    req_frames.extend(value['frames'])

                req_frames = list(set(req_frames))
                self.req_frame_list[vid] = req_frames

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                to_save=self.req_frame_list[video], 
                use_all_mask=True
            )

    def __len__(self):
        return len(self.vid_list)


class VOST_TestDataset:
    def __init__(self, data_root, imset='2017/val.txt', size=-1):
        self.image_dir = path.join(data_root, 'JPEGImages')
        self.mask_dir = path.join(data_root, 'Annotations')
        # self.size_dir = path.join(data_root, 'JPEGImages')
        self.size = size

        with open(path.join(data_root, 'ImageSets', imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)


class MOSE_TestDataset:
    def __init__(self, data_root, split, size=480):
        self.image_dir = path.join(data_root, split, 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations')
        # self.size_dir = path.join(data_root, split, 'JPEGImages')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)


class VISOR_TestDataset:
    def __init__(self, data_root, imset='2017/val.txt', size=-1):
        self.image_dir = path.join(data_root, 'JPEGImages', "480p")
        self.mask_dir = path.join(data_root, 'Annotations', "480p")
        # self.size_dir = path.join(data_root, 'JPEGImages', "480p")
        self.size = size

        with open(path.join(data_root, 'ImageSets', imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)


class BURST_TestDataset:
    def __init__(self, data_root, split, size=480):
        self.image_dir = path.join(data_root, split, 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations_first_only')
        # self.size_dir = path.join(data_root, split, 'JPEGImages')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))
        self.req_frame_list = {}

        with open(path.join(data_root, split, 'first_frame_annotations.json')) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)['sequences']

            for vid in meta:
                assert (vid['dataset'] + '_-_' + vid['seq_name']) in self.vid_list
                req_frames = [i[:-4] for i in vid['annotated_image_paths']]
                self.req_frame_list[str(vid['dataset'] + '_-_' + vid['seq_name'])] = req_frames

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                to_save=self.req_frame_list[video],
                use_all_mask=True,
            )

    def __len__(self):
        return len(self.vid_list)


class LVOS_TestDataset:
    def __init__(self, data_root, split, size=480):
        self.image_dir = path.join(data_root, split, 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations_first_only' if split == 'valid' else 'Annotations')
        # self.size_dir = path.join(data_root, split, 'JPEGImages')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))
        self.req_frame_list = {}

        with open(path.join(data_root, split, f'{split}_meta.json')) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)['videos']

            for vid in self.vid_list:
                req_frames = []
                objects = meta[vid]['objects']
                for object in objects.values():
                    req_frames.append(int(object['frame_range']['start']))
                    req_frames.append(int(object['frame_range']['end']))

                req_frames = list(set(req_frames))
                vid_start, vid_end = min(req_frames), max(req_frames)
                self.req_frame_list[vid] = generate_arithmetic_sequence(vid_start, vid_end, step=5)

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video,
                path.join(self.image_dir, video),
                path.join(self.mask_dir, video),
                size=self.size,
                to_save=self.req_frame_list[video],
                use_all_mask=True,
            )

    def __len__(self):
        return len(self.vid_list)


def generate_arithmetic_sequence(start_str, end_str, step=1, zfill_len=8):
    start_num = int(start_str)
    end_num = int(end_str)

    if start_num > end_num:
        start_num, end_num = end_num, start_num  # 保证 start_num <= end_num

    sequence = [str(i).zfill(zfill_len) for i in range(start_num, end_num + 1, step)]

    return sequence