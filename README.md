# Contrastive-Learning-Cifar10

This repository is reproduced several Self-Supervised Learning (SSL) methods and their comparison.
Including:

1. DINO: https://arxiv.org/abs/2104.14294
2. MoCoV3: https://arxiv.org/abs/2104.02057
3. AdCo: https://arxiv.org/abs/2011.08435
4. BYOL: https://arxiv.org/abs/2006.07733
5. SimCLR: https://arxiv.org/abs/2002.05709


Besides, this repository also includes several Supervised Learning (SL) methods or tricks.
Including:
1. SwinTransformer: https://arxiv.org/abs/2103.14030
2. RedNet (involution): https://arxiv.org/abs/2103.06255
3. ViT: https://arxiv.org/abs/2010.11929
4. SAM optimazation: https://arxiv.org/abs/2010.01412
5. ACON activation: https://arxiv.org/abs/2009.04759
6. EfficientNet: https://arxiv.org/abs/1905.11946
7. PyramidNet: https://arxiv.org/abs/1610.02915
8. AdvProp: https://arxiv.org/abs/1911.09665
9. RandAug: https://arxiv.org/abs/1909.13719
10. MixUp: https://arxiv.org/abs/1710.09412

Note that a lot of codes are borrowed from other repositories. Through my hand, now these codes may like shit.

# How to use
The SL and SSL methods can be directly run by `python main_cifar10.py` and `python train_contrastive_learning_cifar10.py`, respectively.

Before running these codes, you need specify the configuration file in `./config/xxx/xxx.cf`.

You may find some error like "Parameters are missing", that's because as I gradually add new models or tricks,
 the new parameters in the code will be missing in the old configuration files.
 

# Experiment results
In SSL part, the backbone is PyramidNet and ViT, the training logs are in `./results/ContrastiveLearning`, you can use tensorboard to visualize them. 
Here I summarize some results below in table.

Common setting: k=7(kNN-Monitor), lr=2e-1, out_dim=128, bs=256, weight_decay=1e-4, optim=SGD

Methods|Backbone|Config|Temperature|Other param|kNN Top1
:--:|:--:|:--:|:--:|:--:|:--:
SimCLR|PyramidNet|SimCLR_PyramidNet_1.cf|0.2|None|86.46
MoCoV3|PyramidNet|MoCo_PyramidNet_4.cf|0.07|memory_bank=True<br>K=4096|84.98
MoCoV3|PyramidNet|MoCo_PyramidNet_8.cf|0.07|memory_bank=True<br>K=10240|84.55
MoCoV3|PyramidNet|MoCo_PyramidNet_5.cf|0.07|memory_bank=False|83.37
MoCoV3|PyramidNet|MoCo_PyramidNet_6.cf|0.2|memory_bank=False|88.27
MoCoV3|PyramidNet|MoCo_Multicrop_PyramidNet_6.cf|0.2|memory_bank=False<br>local_crops=8|93.79
MoCoV3|PyramidNet|MoCo_PyramidNet_7.cf|0.2|memory_bank=True<br>K=10240|89.46
AdCo|PyramidNet|AdCo_PyramidNet_1.cf|0.12|memory_lr=3.0<br>mem_t=0.02|81.27
AdCo|PyramidNet|AdCo_PyramidNet_2.cf|0.2|memory_lr=1.0<br>mem_t=0.1|86.17
BYOL|PyramidNet|BYOL_PyramidNet_1.cf|None|None|89.22
DINO|ViT|DINO_ViT_1.cf|student_t=0.1<br>teacher_t=0.04|lr=5e-4<br>local_crops=8|83.93
DINO|ViT|DINO_ViT_2.cf|student_t=0.2<br>teacher_t=0.1|lr=5e-4<br>local_crops=8|83.78
DINO|ViT|DINO_ViT_3.cf|student_t=0.1<br>teacher_t=0.04|lr=5e-4|77.92
MoCoV3|ViT|MoCo_ViT_1.cf|0.1|memory_bank=False<br>local_crops=8|72.14
MoCoV3|ViT|MoCo_ViT_2.cf|0.07|memory_bank=False<br>local_crops=8|68.54
MoCoV3|ViT|MoCo_ViT_3.cf|0.2|memory_bank=False<br>local_crops=8|75.75

