# Revisiting Salient Object Detection: Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects

This repository contains code for the paper 

**[Revisiting Salient Object Detection: Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects](http://openaccess.thecvf.com/content_cvpr_2018/papers/Islam_Revisiting_Salient_Object_CVPR_2018_paper.pdf)**,
<br>
Presented at [CVPR 2018](http://cvpr2018.thecvf.com/)

If you find the code useful for your research, please consider citing our work:

    @InProceedings{Islam_2018_CVPR,
      author = {Amirul Islam, Md and Kalash, Mahmoud and Bruce, Neil D. B.},
      title = {Revisiting Salient Object Detection: Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2018}
    }
    
## How to use these codes?

Anyone can freely use our codes for what-so-ever purpose they want to use. Here we give a detailed instruction to set them up and use for different applications.

The codes can be downloaded using the following command:

        git clone --recursive https://github.com/islamamirul/rsdnet.git
        cd rsdnet
**Setup:**

1. Download and compile caffe-rsdnet which is a modified version of [deeplab-public-ver2](https://bitbucket.org/aquariusjay/deeplab-public-ver2.git)

2. Download the PASCAL-S dataset from [here](https://www.dropbox.com/sh/a109tphyadzt1es/AABsovfaxOL7lEqc6ne9PZi3a?dl=0) and put them under ./data/

3. Run the following script to generate stack of saliency masks

        ./scripts/stack_generation/generate_saliency_mask_stack.m
        
4. Download the pretrained-weights (init.caffemodel) of DeepLabv2 from [here](http://liangchiehchen.com/projects/DeepLabv2_resnet.html) and put it under ./models/trained_weights/

## Trained Model
You can download the trained model which is reported in our paper at [Dropbox](https://www.dropbox.com/sh/we3vk0z9nln0jao/AABVOTQ2N9kcBN_gnN2rJ11Wa?dl=0) and put them under ./models/trained_weights/

## Training RSDNet

Modify the caffe root directory and run the following command to start training:

    sh train_rsdnet.sh

## Testing RSDNet

Modify the caffe root directory in ./scipts/inference/test_rsdnet.py and run the following command:

    sh test_rsdnet.sh
    
## Results

The results of multiple salienct object detection (extended to salient object ranking) on PASCAL-S dataset can be found at [Dropbox](https://www.dropbox.com/sh/y5kwsotiqkw4dly/AAB_Fpvv-_ZYlCSPl9A-xIdsa?dl=0)

**Salient Object Ranking (SOR):** Please run the following script to generate overall SOR score for RSDNet
        
           ./scripts/eval/SOR/SOR.m
         
To generate the detection scores (F-measure, AUC, and MAE) please run the corresponding scripts under .scripts/eval/..
 
        
