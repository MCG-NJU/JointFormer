# Dataset

The structure is almost the same as the one in [STCN](https://github.com/hkchengrex/STCN?tab=readme-ov-file#data-preparation) and [XMem](https://github.com/hkchengrex/XMem/blob/main/docs/GETTING_STARTED.md#dataset).


```
├── JointFormer
│ 
├── static
│   ├── BIG_small
│   └── ...
│ 
├── BL30K
│ 
├── DAVIS
│   ├── 2016
│   │   ├── Annotations
│   │   └── ...
│   └── 2017
│       ├── test-dev
│       │   ├── Annotations
│       │   └── ...
│       └── trainval
│           ├── Annotations
│           └── ...
│ 
├── YouTubeVOS-2018
│   ├── all_frames
│   │   └── valid_all_frames
│   ├── train
│   └── valid
│ 
├── YouTubeVOS-2019
│   ├── all_frames
│   │   └── valid_all_frames
│   ├── train
│   ├── train_480p
│   └── valid
│ 
├── MOSE
│    ├── train
│    ├── train_480p
│    ├── valid
│    ├── meta_train.json
│    └── meta_valid.json
│ 
├── LVOS
│    ├── train
│    ├── train_480p
│    ├── valid
│    │   ├── Annotations
│    │   ├── Annotations_first_only
│    │   └── JPEGImages
│    ├── train_meta.json
│    ├── valid_meta.json
│    └── test_meta.json
│ 
├── VOST
│    ├── Annotations
│    ├── ImageSets
│    ├── JPEGImages
│    ├── JPEGImages_10fps
│    └── Videos
│ 
├── BURST
│    ├── frames
│    │   ├── train
│    │   ├── val
│    │   └── test
│    ├── annotations
│    ├── train
│    │   ├── Annotations
│    │   └── JPEGImages
│    ├── val
│    │   ├── Annotations
│    │   ├── Annotations_first_only
│    │   └── JPEGImages
│    └── test
│    │   ├── Annotations
│    │   ├── Annotations_first_only
│    │   └── JPEGImages
│ 
├── VISOR
│    ├── JPEGImages
│    │   └── 480p
│    ├── Annotations
│    │   └── 480p
│    └── ImageSets
│        └── 2022
│            ├── train.txt
│            ├── val.txt
│            └── val_unseen.txt
├── OVIS-VOS-train
     ├── JPEGImages
     └── Annotations
```

### static
- Download Link: https://drive.usercontent.google.com/download?id=1wUJq3HcLdN-z1t4CsUhjeZ9BVDb9YKLd

### BL30K
- Homepage: https://github.com/hkchengrex/MiVOS/#bl30k
- Download Link: https://doi.org/10.13012/B2IDB-1702934_V1

### DAVIS
- Homepage: https://davischallenge.org/
- Download Links: [DAIVS 2016](https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip), [DAVIS-2017-TrainVal-480p](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip), [DAVIS-2017-TrainVal-Full-Resolution](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip), [DAVIS-2017-Test-Dev-480p](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip), [DAVIS 2017-Test-Dev-Full-Resolution](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-Full-Resolution.zip)

```
DAVIS
├── 2016
│   ├── Annotations
│   └── JPEGImages
└── 2017
    ├── test-dev
    │     ├── Annotations
    │     ├── ImageSets
    │     │   └── 2017
    │     │         └── test-dev.txt
    │     └── JPEGImages
    └── trainval
          ├── Annotations
          ├── ImageSets
          │   ├── 2016
          │   │     ├── train.txt
          │   │     └── val.txt
          │   └── 2017
          │         ├── train.txt
          │         └── val.txt
          └── JPEGImages
```

### Youtube-VOS 2018
- Homepage: https://youtube-vos.org/challenge/2018/
- Google Drive: https://drive.google.com/drive/folders/1bI5J1H3mxsIGo7Kp-pPZU8i6rnykOw7f
- Directory structure:
    ```
    YouTubeVOS-2018
    ├── all_frames
    ├ └── valid_all_frames
    ├── train
    └── valid
   ```

### Youtube-VOS 2019
- Homepage: https://youtube-vos.org/challenge/2019/
- Google Drive: https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz
- Resize train as train_480p with [resize_youtube.py](../scripts/resize_youtube.py):
    ```bash
    python ../scripts/resize_youtube.py ../YouTube/train ../YouTube/train_480p
    ```
- Directory structure:
    ```
    YouTubeVOS-2019
    ├── all_frames
    ├ └── valid_all_frames
    ├── train
    ├── train_480p
    └── valid
   ```

### MOSE
- Homepage: https://henghuiding.github.io/MOSE/
- Google Drive: https://drive.google.com/drive/folders/1vChKHzbboP1k6wd6t95guxxURW3nIXBe
- remove unused hidden files
    ```bash
    find ./ -type f -name '._*' -exec rm -f {} \;
    ```
- Resize train as train_480p with [resize_youtube.py](../scripts/resize_youtube.py):
    ```bash
    python ../scripts/resize_youtube.py ../MOSE/train ../MOSE/train_480p
    ```
- Directory structure:
    ```
    MOSE
    ├── train
    │   ├── Annotations
    │   └── JPEGImages
    ├── valid
    │   ├── Annotations
    │   └── JPEGImages
    ├── meta_train.json
    └── meta_valid.json
   ```

### LVOS
- Homepage: https://lingyihongfd.github.io/lvos.github.io/
- Google Drive: [Training Set](https://drive.google.com/file/d/1pdA1Y7-VE4coj6yacya-kolZs6hKuQpS/view?usp=share_link), [Validation Set](https://drive.google.com/file/d/1msjV2AAKROc-UsXh8lUic2gQpsLKfjQ0/view?usp=share_link), [Test Set](https://drive.google.com/file/d/1zp8uqiby3o-2jSjZOqQx4ILh-LLqTz-0/view?usp=share_link), [meta jsons](https://drive.google.com/drive/folders/1fOwGggoYNm_GkZIxs68ptHLk4JNF4Ebq?usp=share_link), [jsons with attributes](https://drive.google.com/drive/folders/1cgIYoIXasw3nx_saYK_M8sb59rtwRFHe?usp=sharing)
- Resize train as train_480p with [resize_youtube.py](../scripts/resize_youtube.py):
    ```bash
    python ../scripts/resize_youtube.py ../LVOS/train ../LVOS/train_480p
    ```
- pre-process it by keeping only the first annotations, provided by [Cutie](https://github.com/hkchengrex/Cutie/blob/main/scripts/data/preprocess_lvos.py)
    ```bash
    python ../scripts/preprocess_lvos.py ../LVOS/valid/Annotations ../LVOS-v1/valid/Annotations_first_only
    ```
- Directory structure:
    ```
    LVOS
    ├── train
    ├── train_480p
    └── valid
     ├── Annotations
     ├── Annotations_first_only
     └── JPEGImages
   ```

### VOST
- Homepage: https://www.vostdataset.org/
- Download Link: https://tri-ml-public.s3.amazonaws.com/datasets/VOST.zip
- Directory structure:
    ```
    VOST
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── JPEGImages_10fps
    └── Videos
   ```


### BURST
- Homepage: https://github.com/Ali2500/BURST-benchmark#dataset-download
- Download Links: [Image sequences](https://motchallenge.net/tao_download.php) as frame/, [Annotations](https://omnomnom.vision.rwth-aachen.de/data/BURST/annotations.zip) as annotations/
- To generate `train-vos` for BURST, use the script `../scripts/convert_burst_to_vos_train.py` which extracts masks from the JSON file into the DAVIS/YouTubeVOS format for training, provided by [Cutie](https://github.com/hkchengrex/Cutie/blob/main/scripts/convert_burst_to_vos_train.py)
    ```bash
    python ../scripts/convert_burst_to_vos_train.py --json_path ../BURST/annotations/train/train.json --frames_path ../BURST/frames/train --output_path ../BURST/train
    python ../scripts/convert_burst_to_vos_train.py --json_path ../BURST/annotations/val/first_frame_annotations.json --frames_path ../BURST/frames/val --output_path ../BURST/val
    python ../scripts/convert_burst_to_vos_train.py --json_path ../BURST/annotations/test/first_frame_annotations.json --frames_path ../BURST/frames/test --output_path ../BURST/test
    ```
- pre-process it by keeping only the first annotations, provided by [Cutie](https://github.com/hkchengrex/Cutie/blob/main/scripts/data/preprocess_lvos.py)
    ```bash
    python ../scripts/preprocess_lvos.py ../BURST/val/Annotations ../BURST/val/Annotations_first_only
    python ../scripts/preprocess_lvos.py ../BURST/test/Annotations ../BURST/test/Annotations_first_only
    ```
- Directory structure:
    ```
    BURST
    ├── train-vos
    └── val-vos
     ├── Annotations
     ├── Annotations_first_only
     └── JPEGImages
    └── test-vos
     ├── Annotations
     ├── Annotations_first_only
     └── JPEGImages
   ```

### VISOR

Homepage: https://epic-kitchens.github.io/VISOR/

1. Create a directory for the dataset
    ```bash
    mkdir ../VISOR && cd ../VISOR
    git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts
    git clone https://github.com/epic-kitchens/VISOR-FrameExtraction
    git clone https://github.com/epic-kitchens/VISOR-VOS
    mkdir VISOR_Sparse && mkdir VISOR_Dense
    ```

2. Download the dataset from [here](https://data.bris.ac.uk/datasets/tar/2v6cgv1x04ol22qp9rm9x2j6a7.zip), and unzip it.

3. Unzip it and sort the files as follows:
    ```python
    import os
    
    # GroundTruth-SparseAnnotations
    root = './VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations/rgb_frames'
    images_root = os.path.join(os.path.dirname(root), 'images_root')
    os.makedirs(images_root, exist_ok=True)
    
    for split in sorted(['train', 'val', 'test']):
        for P in sorted(os.listdir(os.path.join(root, split))):
            if not os.path.isdir(os.path.join(root, split, P)):
                continue
    
            for f_name in sorted(os.listdir(os.path.join(root, split, P))):
                if not f_name.endswith('.zip'):
                    continue
                f_path = os.path.join(root, split, P, f_name)
                print(f'unzip -q {f_path} -d {os.path.join(images_root, f_name[:-4])}')
                os.system(f'unzip -q {f_path} -d {os.path.join(images_root, f_name[:-4])}')
    
    # Interpolations-DenseAnnotations
    root = './VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations'
    annotations_root = os.path.join(root, 'annotations')
    
    for split in sorted(['train', 'val']):
        os.makedirs(os.path.join(annotations_root, split), exist_ok=True)
        for f_name in sorted(os.listdir(os.path.join(root, split))):
            if not f_name.endswith('.zip'):
                continue
            f_path = os.path.join(root, split, f_name)
            print(f'unzip -q {f_path} -d {os.path.join(annotations_root, split)}')
            os.system(f'unzip -q {f_path} -d {os.path.join(annotations_root, split)}')
    ```

4. Download videos (if you need Dense VISOR)
    ```bash
    python epic-kitchens-download-scripts/epic_downloader.py --videos --output-path ./epic-kitchens-download-scripts
    ```

5. VISOR-FrameExtraction (if you need Dense VISOR)
   - Download these 2 sparse jsons and Correcting the interpolations
    ```bash
    # Interpolations correction
    cd VISOR-FrameExtraction
    wget https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations/annotations/train/P02_01.json
    wget https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations/annotations/val/P03_14.json
    cd ../
    
    # Correcting the interpolations
    python VISOR-FrameExtraction/correct_interpolations.py  --input_dir ./VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/annotations/train --output_dir ./VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/annotations/train
    python VISOR-FrameExtraction/correct_interpolations.py  --input_dir ./VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/annotations/val/ --output_dir ./VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/annotations/val/
    ```

    - Modify generate_dense_frames.py L130~135 as:
    ```python
        # update generate_dense_frames.py
        split = sys.argv[1]
    
        if split == 'train':
            json_files_path = './VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/annotations/train/' #'../jsons'
        elif split == 'val':
            json_files_path = './VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/annotations/val/' #'../jsons'
        else:
            raise  NotImplementedError
        
        output_directory = './VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/images_root/' #'../out'
        sparse_rgb_images_root = './VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations/images_root/' # '../Images'
        videos_path = './VISOR/epic-kitchens-download-scripts/EPIC-KITCHENS/videos' #'../videos' 
        output_resolution = (854,480) #original interpolation resolution
        generate_dense_images_from_video_jsons(json_files_path,output_directory,sparse_rgb_images_root,videos_path,output_resolution)
    ```

    - Run frame extraction
    ```bash
    python generate_dense_frames.py train
    python generate_dense_frames.py val
    ```


6. VISOR-VOS

    ```bash
    cd VISOR-VOS
    
    # Sparse: To generate val:
    python visor_to_davis.py -set val -keep_first_frame_masks_only 1  -visor_jsons_root ./VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations/annotations -images_root ./VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations/images_root -output_directory ./VISOR/VISOR_Sparse
    # Sparse: To generate train:
    python visor_to_davis.py -set train -keep_first_frame_masks_only 0  -visor_jsons_root ./VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations/annotations -images_root ./VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations/images_root -output_directory ./VISOR/VISOR_Sparse
    
    
    # https://github.com/epic-kitchens/VISOR-FrameExtraction/issues/7
    cp utils/vis.py utils/vis_dense.py
    cp visor_to_davis.py visor_to_davis_dense.py
    # replace utils/vis_dense.py L#32: mask = np.zeros([480,854],dtype=np.uint8)
    # replace visor_to_davis_dense.py L#4: from utils.vis_dense import *
    # replace visor_to_davis_dense.py L#43: image_name = datapoint["image"]["name"][:-4] + '.jpg'
    # replace visor_to_davis_dense.py: subsequence -> interpolation
    
    # Dense: To generate val:
    python visor_to_davis_dense.py -set val -keep_first_frame_masks_only 1  -visor_jsons_root ./VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/annotations/ -images_root ./VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/images_root -output_directory ./VISOR/VISOR_Dense
    # Dense: To generate train:
    python visor_to_davis_dense.py -set train -keep_first_frame_masks_only 0  -visor_jsons_root ./VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/annotations -images_root ./VISOR/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations/images_root -output_directory ./VISOR/VISOR_Dense
    ```

7. current direct structure
    
   ```
    VISOR
    ├── JPEGImages
    │   └── 480p
    │       ├── {video_id}
    │       ├── {video_id}
    │       └── ...
    ├── Annotations
    │   └── 480p
    │       ├── {video_id}
    │       ├── {video_id}
    │       └── ...
    └── ImageSets
        └── 2022
            ├── train.txt
            ├── val.txt
            └── val_unseen.txt
   ```

## OVIS-VOS-train
- Homepage: https://songbai.site/ovis/
- Download Link: https://drive.google.com/uc?id=1AZPyyqVqOl6j8THgZ1UdNJY9R1VGEFrX