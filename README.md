# Super Resolution of MR via Learning Virtual Parallel Imaging
The Code is created based on the method described in the following paper:  
**Super Resolution of MR via Learning Virtual Parallel Imaging**  
**Author: Cailian Yang, Xianghao Liao, Yifan Liao, Minghui Zhang, Qiegen Liu**  
Date : Nov. 18, 2021  
**The code and the algorithm are for non-comercial use only.**  
**Copyright 2021, Department of Electronic Information Engineering, Nanchang University.**  

## Optional parameters:  
forward_weight: Weight for forward loss.     
epoch: Specifies number of iterations.  

## The training pipeline of SR-LVP. The left part is data preprocessing, and the right part is the forward and reverse process of the reversible network.
<div align="center"><img src="https://github.com/yqx7150/SR-LVP/blob/main/figs/Fig1.jpg"> </div>

## The pipeline of SR-LVP.
<div align="center"><img src="https://github.com/yqx7150/SR-LVP/blob/main/figs/Fig2.jpg"> </div>

## Reconstruction results of Brain dataset and Cardiac dataset for SR-LVP.
<div align="center"><img src="https://github.com/yqx7150/SR-LVP/blob/main/figs/Fig3.jpg"> </div>

## Comparison results of different algorithms on the brain dataset.
<div align="center"><img src="https://github.com/yqx7150/SR-LVP/blob/main/figs/Fig4.jpg"> </div>

# Train
```python
python 2chto12ch_medical_super_resolution_reconstruction_train.py --out_path="./exps/" \
                                                                  --gamma \
                                                                  --task=<task name> \
                                                                  --forward_weight=12
```

# Test
```python
python 2chto12ch_medical_super_resolution_reconstruction_test.py --task=2to12_ch_cross_test \
                                                                 --out_path="./exps/" \
                                                                 --ckpt=<model path>
```

# Acknowledgement
The code is based on [yzxing87/Invertible-ISP](https://github.com/yzxing87/Invertible-ISP)