# UNet Family for Salient Object Detection

This code is mainly revised from UNet++ and more details can be found in this [link](https://github.com/MrGiovanni/UNetPlusPlus).
We modify it into more general binary segmentation applications, like **salient object detection (SOD)** etc.

# Usage

1. Flowing this [instructions](https://github.com/GewelsJI/UNet_Family/blob/master/README_UNetPlusPlus.md) and configuring your virtual environment with python 3, Keras 2.2.2, and Tensorflow-gpu 1.4.1.

2. Training:
    
    - run `cd scripts` and configure your training script in the `MyTrain_XNet.py`

3. Inference:

    - set your testing path and run `MyTest_XNet.py`

# Recommendations

- [2020]\[AAAI]Non-local U-Nets for Biomedical Image Segmentation
    
    - [Tensorflow Official Version](https://github.com/divelab/Non-local-U-Nets)
    - [PyTorch Implementation](https://github.com/Whu-wxy/Non-local-U-Nets-2D-block)

- UNet-like models with PyTorch
    
    - [Unet-Segmentation-Pytorch-Nest-of-Unets](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets)
    - TODO: modify the pytorch-based code to SOD applications

- Paper collection and implementation of UNet-related model
    - [UNet-family](https://github.com/ShawnBIT/UNet-family)

# Ackonwledgements

Thanks all aforementioned contributors in the segmentation field.
