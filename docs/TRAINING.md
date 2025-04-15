# Training

Our training pipeline is follwing [XMem](https://github.com/hkchengrex/XMem/blob/main/docs/TRAINING.md#training).

For example, you can train model with `eight GPUs` as:
```bash
python -m torch.distributed.run --master_port 25763 \
--nproc_per_node=8 \
jointformer/train.py \
--exp_id retrain \
--stage 02
```

- `--nproc_per_node`: the number of training GPUs.
- `--exp_id`: the unique id of training job. The model files and checkpoints will be saved in `./saves/[name containing datetime and exp_id]`.
- `--stage`: our model is trained progressively with different stages, and `--stage 02` means progressively with stage `0` and `2`. Once one stage is finished, the weights will be saved, and the model automatically starts the next training stage by loading the trained weight of latest stage.
- `--load_network`: you can load pretrained models for fast fine-tuning.
- Other setting can be modified in `util/configuration.py`, like batch size, training step, etc.
- The `tensorboard` visualizes the training process.
- Our model is trained on `6 A6000 GPUs (48G)`.

## Stage
- 0: static images;
- 1: BL30K;
- 2: DAVIS 2017 & Youtube-VOS 2019 longer;
- 3: DAVIS 2017 & Youtube-VOS 2019 shorter;
- 4: VOST;
- 5: MOSE;
- 6: LVOS;
- 7: VISOR;

### Training Script

- First, you need to download [pre-trained ConvMAE](https://drive.google.com/file/d/1AEPivXw0A0b_m5EwEi6fg2pOAoDr8C31/view?usp=sharing), and put it in ./checkpoints/.
- We provide pre-trained checkpoint on [s0](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s0.pth) and [s01](https://github.com/MCG-NJU/JointFormer/releases/download/v0.1/jointformer_s01.pth)
- See `train.sh` to train each dataset in details.