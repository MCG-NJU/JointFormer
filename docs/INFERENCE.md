# Inference

- All command-line inference are accessed with `eval.py`
```
python eval.py \
--model [path to model file] \
--output [where to save the output] \
--dataset [which dataset to evaluate on] \
--split [val for validation or test for test-dev] \
--max_memory_frames [max number frame in memory bank] --topk_num [top-k number]
```
- Download our checkpoints by `checkpoints/download_checkpoint.sh`
- Following benchmarks can be evaluated by `eval.sh`.
- Our model is inferenced on one `A6000 GPU (48G)`.

### DAVIS 2017 val

| Training Stage | Training Data | Extra Data (Pre-training) | ckpt | outputs | $\mathcal{J}$ & $\mathcal{F}$ | $\mathcal{J}$ | $\mathcal{F}$ |
| :----: | :----: | :--------: | :----------------------------------------------------------: | :--: | :--: | :--: | :--: |
|s2 | DAVIS 2017 & Youtube-VOS 2019  |  no  | [jointformer_s2.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s2.pth) | [s2-d17_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s2-d17_val.zip) | 89.1 | 85.9 | 92.2 |
|s02 | DAVIS 2017 & Youtube-VOS 2019  |  static  | [jointformer_s02.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s02.pth) | [s02-d17_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s02-d17_val.zip) | 89.7 | 86.7 | 92.7 |
|s012 | DAVIS 2017 & Youtube-VOS 2019  |  static, BL30K | [jointformer_s012.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s012.pth) | [s012-d17_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s012-d17_val.zip) | 90.1 | 87.0 | 93.2 |

- Inference script
```bash
dataset="D17"
split="val"
max_memory_frames="1"
topk_num="60"
```
- Quantitative results: [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation)


### DAVIS 2017 test

| Training Stage | Training Data | Extra Data (Pre-training) | ckpt | outputs | $\mathcal{J}$ & $\mathcal{F}$ | $\mathcal{J}$ | $\mathcal{F}$ |
| :----: | :----: | :--------: | :----------------------------------------------------------: | :--: | :--: | :--: | :--: |
|s2 | DAVIS 2017 & Youtube-VOS 2019  |  no  | [jointformer_s2.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s2.pth) | [s2-d17_test.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s2-d17_test.zip) | 87.0 | 83.4 | 90.6 |
|s02 | DAVIS 2017 & Youtube-VOS 2019  |  static  | [jointformer_s02.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s02.pth) | [s02-d17_test.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s02-d17_test.zip) | 87.6 | 84.2 | 91.1 |
|s012 | DAVIS 2017 & Youtube-VOS 2019  |  static, BL30K | [jointformer_s012.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s012.pth) | [s012-d17_test.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s012-d17_test.zip) | 88.1 | 84.7 | 91.6 |

- Inference script
```bash
dataset="D17"
split="val"
max_memory_frames="1"
topk_num="60"
```
- Quantitative results: [codalab](https://codalab.lisn.upsaclay.fr/competitions/6812)


### Youtube-VOS 2018

| Training Stage | Training Data | Extra Data (Pre-training) | ckpt | outputs | $\mathcal{G}$ | $\mathcal{J}_S$ | $\mathcal{F}_S$ | $\mathcal{J}_U$ | $\mathcal{F}_U$ |
| :----: | :----: | :--------: | :----------------------------------------------------------: | :--: | :--: | :--: | :--: | :--: | :--: |
|s2 | DAVIS 2017 & Youtube-VOS 2019  |  no  | [jointformer_s2.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s2.pth) | [s2-y18_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s2-y18_val.zip) | 86.0 | 86.0 | 91.0 | 79.5 | 87.5 |
|s02 | DAVIS 2017 & Youtube-VOS 2019  |  static  | [jointformer_s02.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s02.pth) | [s02-y18_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s02-y18_val.zip) | 87.0 | 86.2 | 91.0 | 81.4 | 89.3 |
|s012 | DAVIS 2017 & Youtube-VOS 2019  |  static, BL30K | [jointformer_s012.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s012.pth) | [s012-y18_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s012-y18_val.zip) | 87.6 | 86.4 | 91.0 | 82.2 | 90.7 |

- Inference script
```bash
dataset="Y18"
split="val"
max_memory_frames="3"
topk_num="120"
```
- Quantitative results: [codalab](https://codalab.lisn.upsaclay.fr/competitions/7685)


### Youtube-VOS 2019

| Training Stage | Training Data | Extra Data (Pre-training) | ckpt | outputs | $\mathcal{G}$ | $\mathcal{J}_S$ | $\mathcal{F}_S$ | $\mathcal{J}_U$ | $\mathcal{F}_U$ |
| :----: | :----: | :--------: | :----------------------------------------------------------: | :--: | :--: | :--: | :--: | :--: | :--: |
|s2 | DAVIS 2017 & Youtube-VOS 2019  |  no  | [jointformer_s2.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s2.pth) | [s2-y19_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s2-y19_val.zip) | 86.2 | 85.7 | 90.5 | 80.4 | 88.2 |
|s02 | DAVIS 2017 & Youtube-VOS 2019  |  static  | [jointformer_s02.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s02.pth) | [s02-y19_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s02-y19_val.zip) | 87.0 | 86.1 | 90.6 | 82.0 | 89.5 |
|s012 | DAVIS 2017 & Youtube-VOS 2019  |  static, BL30K | [jointformer_s012.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s012.pth) | [s012-y19_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s012-y19_val.zip) | 87.4 | 86.5 | 90.9 | 82.0 | 90.3 |

- Inference script
```bash
dataset="Y19"
split="val"
max_memory_frames="3"
topk_num="120"
```
- Quantitative results: [codalab](https://codalab.lisn.upsaclay.fr/competitions/6066)


### VOST

| Training Stage | Training Data | Extra Data (Pre-training) | ckpt | outputs | $\mathcal{J}$ | $\mathcal{J}_{tr}$ | $\Delta_{tr}$ |
| :----: | :----: | :--------: | :----------------------------------------------------------: | :--: | :--: | :--: | :--: |
|s24 | VOST  |  DAVIS 2017 & Youtube-VOS 2019 | [jointformer_s24.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s24.pth) | [s24-vost_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s24-vost_val.zip) | 51.3 | 35.2 | -16.1 |
|s024 | VOST  |  static, DAVIS 2017 & Youtube-VOS 2019 | [jointformer_s024.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s024.pth) | [s024-vost_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s024-vost_val.zip) | 52.8 | 36.0 | -16.8 |

- Inference script
```bash
dataset="VOST"
split="val"
max_memory_frames="3"
topk_num="30"
```
- Quantitative results: [VOST-evaluation](https://github.com/TRI-ML/VOST/tree/main/evaluation)


### MOSE

| Training Stage | Training Data | Extra Data (Pre-training) | ckpt | outputs | $\mathcal{J}$ & $\mathcal{F}$ | $\mathcal{J}$ | $\mathcal{F}$ |
| :----: | :----: | :--------: | :----------------------------------------------------------: | :--: | :--: | :--: | :--: |
|s5 | MOSE  |  no  | [jointformer_s5.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s5.pth) | [s5-mose_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s5-mose_val.zip) | 66.2 | 62.3 | 70.1 |
|s05 | MOSE  |  static  | [jointformer_s05.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s05.pth) | [s05-mose_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s05-mose_val.zip) | 69.7 | 65.8 | 73.6 |
|s015 | MOSE  |  static, BL30K | [jointformer_s015.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s015.pth) | [s015-mose_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s015-mose_val.zip) | 70.2 | 66.3 | 74.0 |

- Inference script
```bash
dataset="MOSE"
split="val"
max_memory_frames="3"
topk_num="180"
```
- Quantitative results: [codalab](https://codalab.lisn.upsaclay.fr/competitions/10703)


### LVOS val

| Training Stage | Training Data | Extra Data (Pre-training) | ckpt | outputs | $\mathcal{J}$ & $\mathcal{F}$ | $\mathcal{J}$ | $\mathcal{F}$ | $\mathcal{V}\downarrow$ |
| :----: | :----: | :--------: | :----------------------------------------------------------: | :--: | :--: | :--: | :--: | :--: |
|s02 | no  |  static, DAVIS 2017 & Youtube-VOS 2019 | [jointformer_s02.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s02.pth) | [s02-lvos_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s02-lvos_val.zip) | 63.1 | 58.7 | 675. | 31.4 |
|s012 | no  |  static, BL30K, DAVIS 2017 & Youtube-VOS 2019 | [jointformer_s012.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s012.pth) | [s012-lvos_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s012-lvos_val.zip) | 66.6 | 62.1 | 71.1 | 27.9 |
|s026 | LVOS  |  static, DAVIS 2017 & Youtube-VOS 2019 | [jointformer_s026.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s026.pth) | [s026-lvos_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s026-lvos_val.zip) | 63.7 | 59.0 | 68.4 | 29.4 |
|s0126 | LVOS  |  static, BL30K, DAVIS 2017 & Youtube-VOS 2019 | [jointformer_s0126.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s0126.pth) | [s0126-lvos_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s0126-lvos_val.zip) | 67.1 | 62.9 | 71.3 | 29.9 |

- Inference script
```bash
dataset="LVOS"
split="val"
max_memory_frames="3"
topk_num="120"
```
- Quantitative results: [lvos-evaluation](https://github.com/LingyiHongfd/lvos-evaluation)


### LVOS test

| Training Stage | Training Data | Extra Data (Pre-training) | ckpt | outputs | $\mathcal{J}$ & $\mathcal{F}$ | $\mathcal{J}$ | $\mathcal{F}$ | $\mathcal{V}\downarrow$ |
| :----: | :----: | :--------: | :----------------------------------------------------------: | :--: | :--: | :--: | :--: | :--: |
|s02 | no  |  static, DAVIS 2017 & Youtube-VOS 2019 | [jointformer_s02.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s02.pth) | [s02-lvos_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s02-lvos_test.zip) | 59.9 | 55.8 | 64.1 | 27.2 |
|s012 | no  |  static, BL30K, DAVIS 2017 & Youtube-VOS 2019 | [jointformer_s012.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s012.pth) | [s012-lvos_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s012-lvos_test.zip) | 60.4 | 56.5 | 64.2 | 27.5 |
|s026 | LVOS  |  static, DAVIS 2017 & Youtube-VOS 2019 | [jointformer_s026.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s026.pth) | [s026-lvos_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s026-lvos_test.zip) | 60.5 | 56.3 | 64.6 | 27.8 |
|s0126 | LVOS  |  static, BL30K, DAVIS 2017 & Youtube-VOS 2019 | [jointformer_s0126.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s0126.pth) | [s0126-lvos_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s0126-lvos_test.zip) | 61.4 | 57.6 | 65.2 | 28.2 |

- Inference script
```bash
dataset="LVOS"
split="test"
max_memory_frames="3"
topk_num="120"
```
- Quantitative results: [codalab](https://codalab.lisn.upsaclay.fr/competitions/8767)


### VISOR

| Training Stage | Training Data | Extra Data (Pre-training) | ckpt | outputs | $\mathcal{J}$ & $\mathcal{F}$ | $\mathcal{J}$ & $\mathcal{F}$ unseen |
| :----: | :----: | :--------: | :----------------------------------------------------------: | :--: | :--: | :--: |
|s7 | VISOR  |  no  | [jointformer_s7.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s7.pth) | [s7-visor_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s7-visor_val.zip) | 85.1 | 84.6 |
|s07 | VISOR  |  static  | [jointformer_s07.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s07.pth) | [s07-visor_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s07-visor_val.zip) | 85.9 | 85.4 |

- Inference script
```bash
dataset="VISOR"
split="val"
max_memory_frames="1"
topk_num="600"
```
- Quantitative results: [VISOR-VOS](https://github.com/epic-kitchens/VISOR-VOS)


### BURST val

| Training Stage | Training Data | Extra Data (Pre-training) | ckpt | outputs | HOTA-all | HOTA-common | HOTA-uncommon |
| :----: | :----: | :--------: | :----------------------------------------------------------: | :--: | :--: | :--: | :--: |
|s2 | no  | DAVIS 2017 & Youtube-VOS 2019  | [jointformer_s2.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s2.pth) | [s2-burst_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s2-burst_val.zip) | 63.74 | 63.66 | 63.75 |
|s02 | no  | static, DAVIS 2017 & Youtube-VOS 2019  | [jointformer_s02.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s02.pth) | [s02-burst_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s02-burst_val.zip) | 65.49 | 65.93 | 65.37 |
|s012 | no  | static, BL30K, DAVIS 2017 & Youtube-VOS 2019  | [jointformer_s012.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s012.pth) | [s012-burst_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s012-burst_val.zip) | 66.30 | 66.80 | 66.18 |

- Inference script
```bash
dataset="BURST"
split="val"
max_memory_frames="3"
topk_num="30"
```
- Convert output mask to json: [mask_to_burst_json.py](https://github.com/hkchengrex/Cutie/blob/main/scripts/mask_to_burst_json.py)
- Quantitative results: [BURST-benchmark](https://github.com/Ali2500/BURST-benchmark)


### BURST test

| Training Stage | Training Data | Extra Data (Pre-training) | ckpt | outputs | HOTA-all | HOTA-common | HOTA-uncommon |
| :----: | :----: | :--------: | :----------------------------------------------------------: | :--: | :--: | :--: | :--: |
|s2 | no  | DAVIS 2017 & Youtube-VOS 2019  | [jointformer_s2.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s2.pth) | [s2-burst_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s2-burst_test.zip) | 65.93 | 66.36 | 65.84 |
|s02 | no  | static, DAVIS 2017 & Youtube-VOS 2019  | [jointformer_s02.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s02.pth) | [s02-burst_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s02-burst_test.zip) | 66.73 | 65.21 | 67.03 |
|s012 | no  | static, BL30K, DAVIS 2017 & Youtube-VOS 2019  | [jointformer_s012.pth](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s012.pth) | [s012-burst_val.zip](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/s012-burst_test.zip) | 68.05 | 68.40 | 67.98 |

- Inference script
```bash
dataset="BURST"
split="test"
max_memory_frames="3"
topk_num="30"
```
- Convert output mask to json: [mask_to_burst_json.py](https://github.com/hkchengrex/Cutie/blob/main/scripts/mask_to_burst_json.py)
- Quantitative results: [BURST-benchmark](https://github.com/Ali2500/BURST-benchmark)