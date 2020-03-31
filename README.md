# ResDerainGAN
##GAN-Based Rain Noise Removal From Single-Image Considering Rain Composite Models (IEEE Access)
[[Paper Link](https://ieeexplore.ieee.org/document/9016251)] 

Under severe weather conditions, outdoor images or videos captured by cameras can be affected by heavy rain and fog. For example, on a rainy day, autonomous vehicles have difficulty determining how to navigate due to the degraded visual quality of images. In this paper, we address a single-image rain removal problem (de-raining). As compared to video-based methods, single-image based methods are challenging because of the lack of temporal information. Although many existing methods have tackled these challenges, they suffer from overfitting, over-smoothing, and unnatural hue change. To solve these problems, we propose a GAN-based de-raining method. The optimal generator is determined by experimental comparisons. To train the generator, we learn the mapping between rainy and residual images from the training dataset. Besides, we synthesize a variety of rainy images to train our network. In particular, we focus on not only the orientations and scales of rain streaks but also the rainy image composite models. Our experimental results show that our method is suitable for a wide range of rainy images. Our method also achieves better performance on both synthetic and real-world images than state-of-the-art methods in terms of quantitative and visual performances.

![UNetGAN.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/238733/990a2c7f-919d-449c-0ba6-032ad7af0831.png)



<img src="https://qiita-image-store.s3.amazonaws.com/0/238733/dc272717-b7e8-7e3b-a316-9b37daf15fdb.png" width="500">

## Citation

Please cite this paper if you use this code.

```
@ARTICLE{9016251, 
author={T. {Matsui} and M. {Ikehara}}, 
journal={IEEE Access}, 
title={GAN-Based Rain Noise Removal From Single-Image Considering Rain Composite Models}, 
year={2020}, 
volume={8}, 
number={}, 
pages={40892-40900},}
```



## Demo
- De-rain:
`evaluate.py`
- Generating rain noise:
`SynthesizeRainyImage.py`

## Installation
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Contributions
- We explore an optimal deep learning structure for de-raining. Inspired by the success of GANs in other image processing tasks, we introduce a GAN for de-raining. Moreover, we compare the performance with several generator structures. Since a rain detection task needs to capture global features and local features, we conclude that the UNet structure is suitable for de-raining.
- We introduce residual learning to remove rain streaks without losing the textures and edges. We learn the relationship between rainy images and residual images. Compared to a plane network structure which trains the mapping relationship between rainy images and clean images, clearer images are outputted. Also, This speeds up the training process and improves the de-raining performance. For real-world images, we do post-processing to remove haze.
- To create synthetic rainy images, we introduce an automatic rain streaks generator. Almost all rain removal methods use Photoshop to create rain noises. As rain streaks have many parameters, automatically adjusting these parameters is quite difficult. Our proposed method can easily change parameters on Python, which results in saving time and effort to obtain natural rain streaks.
- We propose a combination of two composite models for creating synthetic rainy images. Although most existing methods use only one rain composite model, it is not enough for real-world images. Our experimental results demonstrate that a combination of these models achieves better performance than using either of them.


## Author

[takuro-matsui](https://ieeexplore.ieee.org/author/37086527658)

If you have any questions, please feel free to send us an e-mail matsui@tkhm.elec.keio.ac.jp.
