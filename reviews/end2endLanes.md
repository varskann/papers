## Towards End-to-End Lane Detection: an Instance Segmentation Approach
Davy Neven, Bert De Brabandere, Stamatios Georgoulis, Marc Proesmans, Luc Van Gool, CVPR 2018

### Summary
---
A fast lane detection algorith (~50 fps), able to handle mulitple lanes and lane changes

_tuSimple lane datase_: [http://benchmark.tusimple.ai/#/t/1/dataset](http://benchmark.tusimple.ai/#/t/1/dataset) 

- Lane Detection as an Instance Segmentation(**_LaneNet_**). Two heads: 
	- Lane Segmentation branch : Marking per-pixel as lanes or background. Simple Segmentation task with cross-entropy loss
	- Lane Embedding branch : Label segmented lanes into different lane instances. One-shot method. (Distance base clustering)
	
- Robust perspective transform parameters returned by the network for better lane fitting over varaible road-slopes, etc. (**_H-Net_**). Least Square loss with predicted lane points 


__Note__ : Instead of developing a multiclass detector, divide the task into multiple heads. Each head taking care of simple taks(binary segmentation, classification, etc.)


#### Groun Truth Labeling
_Ground-truth lanes are marked even through objects like occluding cars, or also in the absence of explicit visual lane segments, like dashed or faded lanes._



### Evaluation Metrics
---
- Summation of the Correct points to ground-Truth points.
Lane point is correct if at given _y_, x value is within certain threshold
- FP and FN scores


**_Sweet, Simple paper. Explained each step :)_


