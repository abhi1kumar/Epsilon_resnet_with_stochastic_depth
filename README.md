# Improved ResNets by combining Epsilon-Resnet and Stochastic Depths

This is the course project of the course CS6955: Special Topics in Deep Learning offerred in Fall 2018 at the University of Utah. The project combines the idea of Epsilon Resnet and Stochastic Depth. The code is modified version of [Epsilon-Resnet](https://github.com/yuxwind/epsilonResnet).


### Dependencies
Dependencies is the same as tensorpack

1. Python 2.7 or 3
2. TensorFlow >= 1.3.0

### Installation
Please install the tensorpack libraries by following the steps given below:
```bash
# pull tensorpack
git clone https://github.com/ppwwyyxx/tensorpack.git
# This implementation is based on tags/0.2.0. 
git checkout tags/0.2.0
```


***
### References
```
@inproceedings{yu2018learning,
  title={Learning strict identity mappings in deep residual networks},
  author={Yu, Xin and Yu, Zhiding and Ramalingam, Srikumar},
  booktitle={CVPR},
  year={2018}
}
@inproceedings{huang2016deep,
  title={Deep networks with stochastic depth},
  author={Huang, Gao and Sun, Yu and Liu, Zhuang and Sedra, Daniel and Weinberger, Kilian},
  booktitle={ECCV},
  year={2016}
}
```
