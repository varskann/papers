## Fast R-CNN
Ross Girshick, ICCV 2015

### Summary
---

Project Link: [https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)

Improvement over __R-CNN__, unigying three independent training models into one jointly trained framework and sharing computations.
CNN forward pass is made over the entire image and the regions proposals share the feature matrix and use the same feature matrix for object classification and bounding-box regression

___RoI Pooling___: Max pooling to convert a RoI into a fixed size window. (Input region divided into HxW grids and max pooling is applied on every subwindow, yielding HxW map)

__Steps__:
1. Pre-train a CNN for image classification task
2. Propose Regions by Selective Search(~2K)
3. Alter pre-trained CNN architecture: 
	- Replace last max-pooling layer with the _RoI Pooling_ layer
	- Replace last Fully Connected layer and last softmax layer (K Classes) with a fully connected layer and softmax over K+1 classes
4. Two output branches:
	- A softmax estimator of K+1 classes (K classes + background)
	- A bounding box regression model

__Loss Function__:
Combined loss for two tasks(classificaiton + localization)

For Background, Bounding Box loss is ignored.

```
L = L<sub>cls</sub>(pred, gt) + 1[u>= 1]L<sub>box</sub> {u = 1 for object, u = 0 for background}
```

__Limitations__: 

Region proposals still generated through selective search seperately

---

#### Review
- Focus on slicing of data and time speedup
- New concepts defined(go through code to get more familiarity)
- Insitu work to improce R-CNN, more future work will be in order.
- Explaing all the alternatives tried.
