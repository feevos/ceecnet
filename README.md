# ** UPDATE **
We added a pure semantic segmentation model that uses CEECNetV1, CEECNetV2 or FracTAL ResNet micro-topologies, based on a single encoder/decoder macro-topology (unet-like). These can be found in location: ```models/semanticsegmentation/x_unet```. 

# Looking for change? Roll the Dice and demand Attention
![mantis](images/img_3.png)

Official [mxnet](https://mxnet.incubator.apache.org/) implementation of the paper: ["Looking for change? Roll the Dice and demand Attention"](https://arxiv.org/abs/2009.02062), Diakogiannis et al. (2020). This repository contains source code for implementing and training the mantis ceecnet/FracTAL ResNet as described in our manuscript. All models are built with the mxnet DL framework (version < 2.0), under the gluon api. We do not provide pre-trained weights. 

Inference examples for the task of Building change detection for the model mantis ceecnetV1. From left to right, input image date 1, input image date 2, ground truth, inference, confidence heat map for the segmentation task. 
![mantis](images/img_1.png)
![mantis](images/img_2.png)


### Directory structure: 

```
.
├── chopchop
├── demo
├── doc
├── images
├── models
│   ├── changedetection
│   │   └── mantis
│   ├── heads
│   └── semanticsegmentation
│       └── x_unet
├── nn
│   ├── activations
│   ├── layers
│   ├── loss
│   ├── pooling
│   └── units
├── src
└── utils
```

In directory ```chopchop``` exists code for splitting triplets of raster files (date1, date2, ground truth) into small training patches. It is tailored on the LEVIR CD dataset. In  ```demo``` exists a notebooks that shows how to initiate a mantis ceecnet model, and perform forward and multitasking backward operations. In ```models/changedetection/mantis``` exists a generic definition for arbitrary depth and number of filters, that are described in our manuscript. In ```nn``` exist all the necessary building blocks to construct the models we present, as well as loss function definitions. In particular, in ```nn/loss``` we provide the average fractal Tanimoto with dual (file ```nn/loss/ftnmt_loss.py```), as well as a class that can be used for multitasking loss training. Users of this method may want to write their own custom implementation for multitasking training, based on the ```ftnmt_loss.py``` file. See ```demo``` for example usage with a specific ground truth labels configuration. In ```src``` we provide a mxnet Dataset class, as well as a normalization class. Finally, in utils, there exist a function for selecting BatchNormalization, or GroupNorm, as a paremeter. 


### Datasets 
Users can find the datasets used in this publication in the following locations:  
LEVIR CD Dataset: https://justchenhao.github.io/LEVIR/  
WHU Dataset: http://gpcv.whu.edu.cn/data/building_dataset.html  


### License
CSIRO BSTD/MIT LICENSE

As a condition of this licence, you agree that where you make any adaptations, modifications, further developments, 
or additional features available to CSIRO or the public in connection with your access to the Software, you do so on the terms of the BSD 3-Clause Licence template, a copy available at: http://opensource.org/licenses/BSD-3-Clause.



### CITATION
If you find the contents of this repository useful for your research, please cite:
```
@article{diakogiannis2020looking,
    title={Looking for change? Roll the Dice and demand Attention},
    author={Foivos I. Diakogiannis and François Waldner and Peter Caccetta},
    year={2020},
    eprint={2009.02062},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
