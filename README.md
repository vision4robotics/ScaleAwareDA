# Scale-Aware Domain Adaptation for Robust UAV Tracking

#### Changhong Fu, Teng Li, Junjie Ye, Guangze Zheng, Sihang Li, Peng Lu

This is the official code for the paper "Scale-Aware Domain Adaptation for Robust UAV Tracking". 



## Abstract

Visual object tracking has facilitated diversified autonomous applications when applied on unmanned aerial vehicle (UAV). Most Siamese network (SN)-based trackers are initially trained on general perspective images with rather large objects. However, these trackers are afterwards directly applied to UAV perspective images with relatively small objects. This disparity between the training and inference phases on object scale in a image impairs tracking performance or even entails tracking failure. To tackle such a problem, this work proposes a novel scale-aware domain adaptation (SDA) framework, which is able to adjust existing general trackers and tailor them for UAV applications. Specifically, after constructing datasets with UAV-specific attributes in a simple yet effective way as the target domain, SDA utilizes a Transformer-based feature alignment module to bridge the gap between features in the source domain and target domain. Furthermore, inspired by the contrastive learning strategy, feature projection and prediction modules are carefully designed for the network training. Consequently, trackers trained on general perspective images can represent objects in UAV scenarios more powerfully and thus maintain their robustness. Extensive experiments on three challenging UAV benchmarks have demonstrated the superiority of the SDA framework. In addition, real-world tests on several challenging sequences further attest to the practicality.
![workflow](https://github.com/vision4robotics/ScaleAwareDA/blob/main/image/workflow.png)



## 1. Environment setup

This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2. Please install related libraries before running this code:

```
pip install -r requirements.txt
```



## 2. Test

Tack BAN-SDA for instance:

1.  Download our pretrained model for [BAN-SDA](https://drive.google.com/file/d/1UcynZP6ujc8cEbnO9FfYyUqCMAzp913Q/view?usp=sharing) (or [CAR-SDA](https://drive.google.com/file/d/17cZXfqpm3_xfBxzTFe2yCTX85L20AjHE/view?usp=sharing)) and place it at `SDA-master/BAN-SDA/tools/snapshot` directory.
2.  Download testing datasets and put them into `test_dataset` directory. 

```
python ./tools/test.py                                
	--dataset UAVTrack112                  
    --tracker_name BAN-SDA
	--snapshot snapshot/bansda_model.pth
```

The testing result will be saved in the `results/dataset_name/tracker_name` directory. 



## 3. Train

Tack BAN-SDA for instance:

1. Download the training datasets [VID](https://image-net.org/challenges/LSVRC/2017/), [DET](https://image-net.org/challenges/LSVRC/2017/) and [YouTube](https://pan.baidu.com/share/init?surl=ZTdfqvhIRneGFXur-sCjgg) (code: t7j8).
2.  Download the pre-trained model [SiamBAN](https://drive.google.com/file/d/1SJwPUpTQm6xL44-8jLvDrSMhOzVsbLAZ/view) (or [SiamCAR](https://pan.baidu.com/share/init?surl=ZW61I7tCe2KTaTwWzaxy0w) code: lw7w) and place it at `SDA-master/BAN-SDA/tools/snapshot`  directory.
3. start training

```
cd `SDA-master/BAN-SDA/tools
export PYTHONPATH=$PWD
python train.py
```



## 4. Evaluation

Tack BAN-SDA for instance:

1.  Start evaluating

```
cd SDA-master/BAN-SDA/tools
python eval.py --dataset UAVTrack112
```



## Demo Video



## Contact

If you have any questions, please contact Teng Li at [tengli0204@gmail.com](mailto:tengli0204@gmail.com) or Changhong Fu at [changhongfu@tongji.edu.cn](mailto:changhongfu@tongji.edu.cn). 



## Acknowledgements

We would like to express our sincere thanks [SiamBAN](https://github.com/hqucv/siamban) and [SiamCAR](https://github.com/ohhhyeahhh/SiamCAR) for their efforts. 
