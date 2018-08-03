### Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1
Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio, 2016

---

#### Summary
Method to train a Neural Network with _Binary Weight and Activations_. Drastically reduce memory size and memory accesses and replace most arithmetic operations with bit-wise operations
- Programmed a binary matrix multiplication GPU kernel which runs 7 times faster for MNIST BNN

- [Theano code](https://github.com/MatthieuCourbariaux/
BinaryNet)
- [Torch code](https://github.com/itayhubara/BinaryNet)



#### Binared Neural Networks:
1. **Binarization functions**
	![ Binarization Functions](https://github.com/varskann/papers/blob/master/images/binarization_fn.png  "Binarization Functions")

	Stochastic binarization is more appealing but is harder to implement, hence Deterministic binarization is used in experiments


2. **Gradient Computation**: For _Gradient_ computation actual weights/activations are used as the back prop already works on noisy steps, hence important to keep  necessary resolution

3. **Back Propagation**: For backpropagating through the binarization functions, we assume it to be _'straight-through estimator'_, i.e., ```g<sub>r</sub> = g<sub>q</sub>**1**<sub>|r|<1</sub>

	This preserves the gradient's information and cancels the gradient when _r_ is too large



4. **Shift based Batch Normalization and AdaMax**: to speed up the computations
![Shift based Batch Normalization](https://github.com/varskann/papers/blob/master/images/ShiftBased_BatchNorm.png "Shift based batch normlization") 

![Shift based AdaMax](https://github.com/varskann/papers/blob/master/images/ShiftBased_AdaMax.png  "Shift based AdaMax")

_**Notes**_
1. For a _k_ size kernel, number of unique convolution features is 2<sup>k<sup>2</sup></sup>
2. On MNIST, Cifar-10 and SVHN dataaset, near state-of-art results
3. Training convergence time increases but eventually reaches the same loss/accuracy
4. Key operation of deep learning (Multiply-Accumulate) is replaced by 1-bit XNOR count operations