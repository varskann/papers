## HarDNet: A Low Memory Traffic Network

Ping Chao, Chao-Yang Kao, Yu-Shan Ruan, Chien-Hsiang Huang, Youn-Long Lin, CVPR 2019

Python: https://github.com/PingoLH/Pytorch-HarDNet



### Summary

A harmonic densely connected architecture to reduce the Memory traffic and MACs (Multiply-Accumulate Operations / Floating point operations). 

Introducing CIO as approximation for DRAM(Dynamic Random-Access memory) traffic

Can shrink the size of feature maps but very few papers address lossless compression of feature maps.

	- Quantization methods including sub-sampling reduce DRAM but penalize the accuracy

Paper tried to reduce the DRAM traffic by careful design of the CNN



**Tools used**

1. NVIDIA profiler to measure the number of DRAM read/write bytes for the GPU
2. ARM Scale Sim for mobile devices

**Contributions**

1. New Metric CIO (Convolutions Input/output), as approximation for the DRAM traffic(proportional)

   ![CIO](https://github.com/varskann/papers/blob/master/images/cio.png)

   * If input tensor is a concatenation,  it can be counter multiple time


2. A layers with Computational Density (MoC = MACs / CIO ) below a certain, platfpom-dependent, value has fixed latency. Hence paper tries to constraint the low MoC and avoid using layers with high CIO, e.g., 1x1 Conv

   ![Moc](https://github.com/varskann/papers/blob/master/images/moc.png)

3. Harmonic Sparsification of the DenseNet. Layer k can be connected to layer k-2^n if 2^n divides k (n >= 0; k-2^n >= 0). Once a layer 2^n is processed, layers from 1 through 2^n -1 can be flushed from the memory


### Related Works:

1. To cope up with Degradataion problem, Highway networks and Residual networks sum pu mulitple preceeding layers.
2. Stochastic depth regularization crosses layers that are randomly dropped
3. DenseNets concatenates all preceeding layers as a shortcut achieving more efficient deep supervision
4. SparseNet and LogDenseNet create sparse connections s.t. input channelr numbers decrease from O(L^2) to O(LlogL) but to compensate for the the loss in accuracy by significantly increasing the growth rate(output channel width) , which in turn compensates for the CIO reduction offered by the sparse connections
   1. Layers with higher powers of 2 are more influential
   2. Increase the channel ratio for these layers to avoid a low MoC
5. 