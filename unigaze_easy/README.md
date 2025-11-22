# UniGaze Easy Loader

A tiny, dependency-light **Python package** to load **UniGaze** pretrained models from the Hugging Face Hub.

> One-liner:
```python
import unigaze as ug
model = ug.load("unigaze_h14_joint", device="cuda")  # auto-downloads from HF on first use
```



## 📦 Installation

> This package **does not** install PyTorch for you. Install a matching PyTorch first.

**CPU example**
```bash
## E.g., CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install unigaze
```

