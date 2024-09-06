# ViP
This is the official code for our MICCAI 2024 (Early Accepted) paper:
> [Aligning Medical Images with General Knowledge from Large Language Models](https://arxiv.org/pdf/2409.00341)
> Xiao Fang*, Yi Lin*, Dong Zhang, Kwang-Ting Cheng, Hao Chen


### Requirement
We use Python 3.9 and CUDA 11.7.
```
# Clone the following repository
git clone https://github.com/KaiyangZhou/Dassl.pytorch

# Install torch
torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install dependencies
pip install -r requirments.txt

# Install Dassl library
cd Dassl.pytorch
python setup.py develop
```

### Data preparation
Pneumonia: Please download data [here](https://data.mendeley.com/datasets/rscbjbr9sj/3). We use the chest xray part. The data should be put in the following structure: 
```
|-- /DATA/Pneumonia/chest_xray
|  |-- train
|      |-- normal lung
|          |-- NORMAL-28501-0001.jpeg
|          |--...
|      |-- pneumonia
|          |-- BACTERIA-7422-0001.jpeg
|          |--...
|  |-- test
|      |-- normal lung
|          |-- NORMAL-4512-0001.jpeg
|          |--...
|      |-- pneumonia
|          |-- BACTERIA-40699-0001.jpeg
|          |--...
```

Derm7pt: Please download data [here](https://derm.cs.sfu.ca/Welcome.html). We follow [this paper](https://github.com/CristianoPatricio/coherent-cbe-skin) to split the data. The data should be put in the following structure:
```
|-- /DATA/Derm7pt/image
|   |-- train
|       |-- melanoma
|           |-- Aal002bis.jpg
|           |--
|       |-- nevus
|           |-- Aal012.jpg
|           |--
|   |-- val
|       |-- melanoma
|           |-- Ael490.jpg
|           |--
|       |-- nevus
|           |-- Aal004.jpg
|           |--
|   |-- test
|       |-- melanoma
|           |-- Aal002.jpg
|           |--
|       |-- nevus
|           |-- Aal008.jpg
|           |--
```
We also provide the data split in the DATA folder.

### Training & Evaluation
We provide the shell scripts for training and evaluation.   
Pneumonia:
```
bash scripts/ViP/main_pneumonia.sh
```
Derm7pt:
```
bash scripts/ViP/main_derm.sh
```

## Citation
Pleae cite the paper if you use the code.
```
@article{fang2024aligning,
  title={Aligning Medical Images with General Knowledge from Large Language Models},
  author={Fang, Xiao and Lin, Yi and Zhang, Dong and Cheng, Kwang-Ting and Chen, Hao},
  journal={arXiv preprint arXiv:2409.00341},
  year={2024}
}
```

## Acknowledgment
The code is built on [CoOp](https://github.com/KaiyangZhou/CoOp), thanks for their amazing work!