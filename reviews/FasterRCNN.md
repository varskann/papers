## Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, NIPS 2015

_Project Link:_<br/>
MATLAB: [https://github.com/ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)<br>
Python: [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)<br>
<br>
For a good explanation read [Faster RCNN](https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/)

### Summary
Replace Selecetive Search in Fast RCNN with Region Proposal Layer. RPN is a kind of fully convolutional network, and can be trained end-2-end for the purpose of generating region proposals. (Only RPN usecases: Single Class detectors, Face detection etc.)


Achieves near real-time inference using very deep networks

Steps:

Training is alternate between fine-tunig te region proposalsand then fine-tuning the object detection, while keeping the proposals fixed

1. Use a pretrained CNN(VGG/ResNet/ZFNet) to generate a convolution feature map.
2. Use RPN layer to generate _Region Proposals_ with different anchor settings (9 in total, 3 scales, 3 aspect ratios)
3. Use the aforementioned feature map to get RPN generated region features. Region proposal/Object detection is translation invariant
4. Train RPN layer:
	- Loss-1: Objectess of a proposed region (If the proposed region has > 50% overlap with any of the ground truth objects)
	- Loss-2: Bounding Box Regression Loss
5. Use ROI layer to convert the region feature maps to a fixed size to be passed for bounding box classification


### Related Works:
1. **Object Proposals**: 
	- Grouping of superpixels(Selective Search, CPMC, MCG)
	- Sliding window(Objectness in window, EdgeBoxes)
2. **Object Detection/Segmentaion**: 
	- DeepMask(Segmentaion)
	- MultiBox, OverFeat(Detection)

### Implementaions using Faster RCNN:
- Image Segmentation
- Image Captioning
- 3D object Detection