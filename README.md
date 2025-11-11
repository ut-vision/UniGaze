# [WACV 2026] UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training 
> [[arxiv]](https://arxiv.org/pdf/2502.02307), [[project page]](https://ut-vision.github.io/UniGaze/), [[online demo]](https://huggingface.co/spaces/UniGaze/UniGaze) [[Huggface paper page]](https://huggingface.co/papers/2502.02307)

<a href="https://jqin-home.github.io/">Jiawei Qin</a><sup>1</sup>, 
<a href="https://www.ccmitss.com/zhang">Xucong Zhang</a><sup>2</sup>, 
<a href="https://www.yusuke-sugano.info/">Yusuke Sugano</a><sup>1</sup>, 

<sup>1</sup>The University of Tokyo, <sup>2</sup>Delft University of Technology 


<img src="./teaser.png" width="800" alt="grid"/>



<!-- <h4 align="left">
<a href="">Project Page</a>
</h4> -->



## Overview
This repository contains the official PyTorch implementation of both **MAE pre-training** and **unigaze**.



### Todo:
- :white_check_mark: Release pre-trained MAE checkpoints (B, L, H) and gaze estimation training code.
- :white_check_mark: Release UniGaze models for inference.
- :white_check_mark: Code for predicting gaze from videos
- :white_check_mark: (2025 June 08 updated) Release the MAE pre-training code.
- :white_check_mark: (2025 August 25 updated) Online demo is available.

---


## Installation


To install the required dependencies, run:
```bash
pip install -r requirements.txt
```


## Pre-training (MAE)
Please refer to [MAE Pre-Training](./MAE/README.md).


## Training (Gaze Estimation)

For detailed training instructions, please refer to [UniGaze Training](./unigaze/README.md).

---

## Usage of UniGaze


### Available Models

We provide the following trained models:

|   Filename   | Backbone |   Training Data   | Checkpoint |
|--------------|----------|-------------------|------------|
|`unigaze_b16_joint.pth.tar`  | UniGaze-B | *Joint Datasets* | [Download (Google Drive)](https://drive.google.com/file/d/1xdPbzAX8d3cPAMChFjRThryIWVp9Ng_f/view?usp=sharing) |
|`unigaze_L16_joint.pth.tar`  | UniGaze-L | *Joint Datasets* | [Download (Google Drive)](https://drive.google.com/file/d/1JR20_iGTU8pSXtKIC-_swiSRImWLAbBC/view?usp=sharing) |
|`unigaze_h14_joint.pth.tar`  | UniGaze-H | *Joint Datasets* | [Download (Google Drive)](https://drive.google.com/file/d/16z_Y8_yi53xTw_-5Pw9H4jjAOPebIFdA/view?usp=sharing) | 
|`unigaze_h14_cross_X.pth.tar`| UniGaze-H |  ETH-XGaze       | [Download (Google Drive)](https://drive.google.com/file/d/1BVYGOK5NwXUPr63DnbYGeQ_yqlevv9VR/view?usp=sharing) |




### Loading Pretrained Models
- You can refer to [load_gaze_model.ipynb](./unigaze/load_gaze_model.ipynb) for instructions on loading the model and integrating it into your own codebase.
  - If you want to load the MAE, use `custom_pretrained_path` arguments.
  - If you want to load the UniGaze (MAE + gaze_fc), directly use `load_state_dict`

```python
## Loading MAE-backbone only - this will not load the gaze_fc
mae_h14 = MAE_Gaze(model_type='vit_h_14', custom_pretrained_path='checkpoints/mae_h14/mae_h14_checkpoint-299.pth')

## Loading UniGaze
unigaze_h14_crossX = MAE_Gaze(model_type='vit_h_14') ## custom_pretrained_path does not matter because it will be overwritten by the UniGaze weight
weight = torch.load('logs/unigaze_h14_cross_X.pth.tar', map_location='cpu')['model_state']
unigaze_h14_crossX.load_state_dict(weight, strict=True)
```



#### Predicting Gaze from Videos
To predict gaze direction from videos, use the following script:

```bash
projdir=<...>/UniGaze/unigaze
cd ${projdir}
python predict_gaze_video.py \
    --model_cfg_path configs/model/mae_b_16_gaze.yaml  \
    -i ./input_video \
    --ckpt_resume logs/unigaze_b16_joint.pth.tar
``` 


---

## Citation
If you find our work useful for your research, please consider citing:

```
@article{qin2025unigaze,
  title={UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training},
  author={Qin, Jiawei and Zhang, Xucong and Sugano, Yusuke},
  journal={arXiv preprint arXiv:2502.02307},
  year={2025}
}
```
We also acknowledge the excellent work on [MAE](https://github.com/facebookresearch/mae.git).

---

## License:

This model is licensed under the ModelGo Attribution-NonCommercial-ResponsibleAI License, Version 2.0 (MG-NC-RAI-2.0);
you may use this model only in compliance with the License.
You may obtain a copy of the License at

https://github.com/Xtra-Computing/ModelGo/blob/main/MGL/V2/MG-BY-NC-RAI/LICENSE

A comprehensive introduction to the ModelGo license can be found here: https://www.modelgo.li/

---

## Beyond human eye gaze estimation:
Our method also works for different "faces":

<img src="./gaze_cat_dog.png" width="800" alt="grid"/>

---

## Contact
If you have any questions, feel free to contact Jiawei Qin at jqin@iis.u-tokyo.ac.jp.
