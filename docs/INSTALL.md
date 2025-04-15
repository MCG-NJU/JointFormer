# Install

Our codebase is built upon CUDA 11.3, Python 3.8, PyTorch 1.11.0.

```
conda create -n jointformer python=3.8
conda activate jointformer

# PyTorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# VOS dependencies
pip install opencv-python
pip install -r requirements.txt

# Transformer dependencies
pip install einops timm==0.4.9
```