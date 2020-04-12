In this implementation, the idea of Stochastic Depth with p_L=0.2 was used with our own custom learning rate policy for different epochs. It should be emphasized that the Adaptive Learning rate due to Xin Yu et al. (used in Epsilon ResNet) is NOT used in this implementation.

Dependencies
======================================
Dependencies is the same as tensorpack

1. Python 2.7 or 3
2. TensorFlow >= 1.3.0

Installation
==========================================================================
Please install the tensorpack libraries by following the steps given below:
```bash
# pull tensorpack
git clone https://github.com/ppwwyyxx/tensorpack.git
# This implementation is based on tags/0.2.0. 
git checkout tags/0.2.0
```

Run
===============
```bash
python cifar10_StochasticDepth_EpsilonResnet.py --gpu 0 -n 18 -e 2.5 -o cifar10-e_2.5-n_18 --cifar10
```

In the above command,
```-o``` option indicates the name of the Output Directory of this program.
```-n``` option indicates the number of ResNet blocks (which corresponds to 110 layer ResNet)
