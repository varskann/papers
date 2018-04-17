## Rich feature hierarchies for accurate object detection and semantic segmentation
Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik, CVPR 2014

### Summary
_**R-CNN** short for 'Region  based Convolutional Neural Networks'_

_Project Link:_  [https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)

#### Workflow
- Pre-train a CNN for classification task. Fine-tuned based on AlexNet
- Category independent _Regions of Interest_ using _**Selective Search**_(~2K per image)
- Region proposals diluted(extended) to have 16 pixels of context from the original image and warped to a fix 227x227 size required by the CNN
- Fine-tune CNN on warped proposal regions for N+1(background) classes. 
- Generate a 4096 length feature vector using the CNN and pass the vector to a pre-trained SVM for the classification. SVM is trained per class independently, using proposed regions with _IoU > 0.3_ as positive samples
- Bounding Box Regression to reduce the localization error

#### Advantages
- Works significantly better compared to previous available approacehes(Overfeat, DPM, etc.)
- Benign work, too much scope for imporvement. (Increasing the CNN architecture improves the performance significantly, but time increases 7x, hence a future nut to crack)

#### Caviets
- Running Selective Search even in fast-mode is relatively slow
- Region proposal generated at image level, and feature vetcor generation will be slow at image level
- Three models running sequentially without any shared computation. 

---
### Paper Review
- Slice and Dice of results and false positives(Gives signficant insight of the data, whole pipeline, its perks and caviets)
- _**Focuses on Visualizing Passing and Failing cases to explore the possibilities**_
- Appendix explains the used methods in details, and experiments involved to pick a said approach/other alternatives
- ** Hard Negative Mining** from the valiadtion data.
- Explaining effects with and without utilizing a technique

