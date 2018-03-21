## Net2Net: ACCELERATING LEARNING VIA KNOWLEDGE TRANSFER
Tianqi Chen, Ian Goodfellow, and Jonathon Shlens, ICLR 2016



### Summary
---

Proposes _function preserving transformations_ between newural network specifications. 
Aim : To significantly reduce model training time.  _Transfer knowledge_ from previous best model and improve on top of that, without starting from scratch every time one experimeents with deep architectures
Net2Net : Process of training a _student_ network leveraging knowledge from a _teacher_ network that was already trained on the same task.
Core Idea: Initialize new student network to represent the same function as teacher network, but using different parameters(typically, 1/10 learning rate, as student net is supposed to continue training from the point teacher network ended).

- Net2WiderNet
	- Replace a layer in the network with a wider network. e.g., More convolution channels in CNN
	
	- Initial __units__(till # of units in _teacher_ model) are coppied exactly to the student network. New extra units are randomly sampled from the already intialised weights. Incoming weights to a unit are simply copied along with the unit, while outgoing ones are divided by the number of replicas of that unit so that the output at the next layer remains the same
	
	- ** Not a fully general algorithm yet. Remapping was done manually for the inception architecture, need a _remapping inference algorithm_ to make the remapping functions consistent
	
	- Use a small noise or dropout/some randomisation to encourage identical units to learn different functions
	
- Net2DeeperNet
	- Replaces a layer with two layers, one being initialized to an identity matrix so that the output at the next layer is same
	
	- Works if activation function f satisfies f(If(x)) = f(x) for example ReLU, but not sigmoid, tanh.
	
	

### Experiments
---

- Experimetns on ImageNet using Inception-BN network

- Net2WiderNet accelerates training of a standard Inception network significantly by initializing it with a smaller network

- Making the model both, widder and deeper(~sqrt(2) times), increased accuracy to a new state of the art 78.5% on ImageNet validation. New bigger model converes way faster than the original model.



### Advantages 
---

- Larger newtork immediately performs as well as original network, and any change/training on top of it is guaranteed to be an improvement, so long as each local step is an improvement

- Useful in production systems which essentially have to be lifelong learning systems. Net2Net presents an easy way to immediately shift to a model of higher capacity and reuse trained networks.
 
 - ___Explore the design space / experiments faster and advance the results efficiently___
 
 