## VPGNet: Vanishing Point Guided Network for Lane and Road Marking Detection and Recognition
{Seokju Lee, Junsik Kim, Jae Shin Yoon, Seunghak Shin, Oleksandr Bailo, Namil Kim, Tae-Hee Lee, Hyun Seok Hong, Seung-Hoon Han, In So Kweon}, ICCV 2017


### Summary
---
Project Link : [https://github.com/SeokjuLee/VPGNet](https://github.com/SeokjuLee/VPGNet) 
YouTube : [https://www.youtube.com/watch?v=jnewRlt6UbI](https://www.youtube.com/watch?v=jnewRlt6UbI) 

__AIM__ : An end-to-end framework, to achieve high robustness in various weather conditions. Focusing on advancing Autonomous Driving Research

__Contributions__
1. Dataset for _Lane and Road Marking Detection and Recognition Benchmark_. ~20000 annotated images with Vanishing Point and 17 classes of lanes and road markings from around South Korea
2. End-to-end multi-task network for lane and road marking detection and recognition guided by the _vanishing point_. Focusing on adverse weather conditions(night, rainy, etc.)
3. Grid level annotation of lanes and road markings. Image divided into grids of size 8x8 and the grid cell is filled with a class label if any pixel from the original annotation lies within the grid cell
4. Four tasks assigned to the network: 
	- Grid Box Regression
	- Object Detection
	- Multi-label classification
	- *__Vanishing Point Detection__* : VP can be used to provide global geometric context of a scene, which is used to infer lanes and road markings. Image divided into _4 quadrants_ based on the VP location and every pixel will belong to one of the quadrants(intersection of 4 quadrants is the VP) or the absence quadrant in case no VP is present.
	
- *__Lanes__*: Subsample local peaks from multi-label task with high lane probability. Selected points projected to birds-eye view and separate points near the VP. Density based(pixel distance) based clustering, sorting clusters based on vertical index. Qudratic regression of the lines from the obtained clusters and location of VP.

- *__Road Markings__*: Extract grid cells from grid regression task with high-confidence for each class from multi-label output. Merge nearby grids belonging to the same class.



<br/>
### Conclusion & Evaluation
---
1. More Tasks trained, more neurons respond.

2. Lanes Evauation: Measure distance between points on predicted lane to center of the ground truth grid cell for that lane. Within certain boundary R, sample points on the lane are considered true positive. plus F1 score

3. Road Markings : Number of overlapping grid cells in the predicted and ground truth marking blob. plus Recall 

4. Vanishing Point : Euclidean distance between ground truth and predicted VP

5. Grid level annotaion generated stronger grandients fromt the edge information  around thinly annotated areas, hence better performance.
	
5. VP prediciton improves lane and road marking detection performance, and vice-versa
	
