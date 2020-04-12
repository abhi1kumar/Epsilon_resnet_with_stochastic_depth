
import numpy as np
import argparse
import os
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../../../tensorpack')
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.gradproc import SummaryGradient

from EpsilonResnetBase import *
from compressModel import read_cfg

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

BATCH_SIZE = 128
EPSILON = 2.5
NUM_UNITS = None
IS_CIFAR10 = True
NUM_CLASS = 10

class Model(ModelDesc):

    def __init__(self, EPSILON, NUM_CLASS, n):
        super(Model, self).__init__()
        self.n = n
        self.EPSILON = EPSILON
        self.NUM_CLASS = NUM_CLASS
        self.p_l = 0.2 # Changed to 0.2 instead of 0.5
        self.k = 0
        self.is_training = True
        print("Inside the constructor of Class Model")

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
       
        self.is_training = get_current_tower_context().is_training

        image, label = inputs
        image = image / 128.0
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])
        
        all_cnt = tf.constant(self.n * 3+2, tf.float32, name="all_cnt")
        preds = []

        epsilon = get_scalar_var('epsilon', self.EPSILON, summary=True)

        def residual_convs(l, first,out_channel,stride1):
            b1 = l if first else BNReLU(l)
            c1 = Conv2D('conv1', b1, out_channel, stride=stride1, nl=BNReLU)
            c2 = Conv2D('conv2', c1, out_channel)
            return c2
                
        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[1]

            block_type = name[3];
            self.k = self.k + 1

            if (first):
                p=0.0
            else:
                numerator = self.k
                p = numerator * float(1-self.p_l)/(3*self.n)    

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
                short_cut = AvgPooling('pool', l, 2)
                short_cut = tf.pad(short_cut, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])
            else:
                out_channel = in_channel
                stride1 = 1
                short_cut = l

            with tf.variable_scope(name) as scope:

                if self.is_training:
                    print("Inside the Training Block in Model CLASS")
                    means = p
                    mask  = tf.where((tf.random_uniform([1]) - means < 0), tf.ones([1],dtype=tf.float32), tf.zeros([1],dtype=tf.float32), 'mask')
                    invrted_drop_out = 1/((1-means)+0.000001)

                    is_dropped = tf.nn.relu(tf.reduce_sum(mask))
                    is_dropped = tf.where(tf.equal(is_dropped, 1.0), 1.0, 0.0, 'is_dropped')
                    add_moving_summary(is_dropped)
    
                else:
                    print("Inside the Test Block in Model CLASS")
                    mask = 0;
                    invrted_drop_out = 1


                l = residual_convs(l, first, out_channel, stride1)
    
                l = mask * short_cut + (1.0 - mask) * l
                identity_w = (1.0 - mask) * strict_identity(l, self.EPSILON)
                identity_w = tf.nn.relu(tf.reduce_sum(identity_w))
                l = identity_w * invrted_drop_out * l + short_cut
                is_discarded = tf.where(tf.equal(identity_w, 0.0), 1.0, 0.0, 'is_discarded')
                preds.append(is_discarded)
                add_moving_summary(is_discarded)

            return l
            
        side_output_cost = []
        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='NCHW'), \
                argscope(Conv2D, nl=tf.identity, use_bias=False, kernel_shape=3,
                         W_init=variance_scaling_initializer(mode='FAN_OUT')):
            l = Conv2D('conv0', image, 16, nl=BNReLU)
            l = residual('res1.0', l, first=True)
            for k in range(1, self.n):
                l= residual('res1.{}'.format(k), l)
            # 32,c=16
            
            l = residual('res2.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res2.{}'.format(k), l)
                if k == self.n/2:
                    side_output_cost.append(side_output('res2.{}'.format(k), l, label, self.NUM_CLASS))
            # 16,c=32
            l = residual('res3.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res3.' + str(k), l)
            l = BNReLU('bnlast', l)
            # 8,c=64
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, out_dim=self.NUM_CLASS, nl=tf.identity)
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
        
        discarded_cnt = tf.add_n(preds, name="discarded_cnt")
        discarded_ratio = tf.divide(
                discarded_cnt, all_cnt, name="discarded_ratio")
        add_moving_summary(discarded_cnt, discarded_ratio) 
        
        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        side_loss_w = [0.1]
        side_output_cost = [tf.multiply(side_loss_w[i], side_output_cost[i])\
                for i in range(len(side_output_cost))]
        loss = side_output_cost + [cost, wd_cost] 
        self.cost = tf.add_n(loss, name='cost')
        
    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.90)
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    if IS_CIFAR10:
        print('train on cifar10')
        ds = dataset.Cifar10(train_or_test)
    else:
        print('train on cifar100')
        ds = dataset.Cifar100(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config(out_dir):
    print("outdir: %s"%out_dir)
    logger.set_logger_dir('train_log.' + out_dir)
    dataset_train = get_data('train')
    dataset_test = get_data('test')
    MAX_EPOCH = 400
    side_layers = ['res2.{}'.format(int(NUM_UNITS/2))]
    side_prediction_name = ["side_output/" +x for x in side_layers]
    side_inferences = [ClassificationError(x+ "/incorrect_vector",\
            x + "/val_error") for x in side_prediction_name]
    inferences = side_inferences + [ScalarStats('cost'), ClassificationError()]
    print("Calling the Training Method")
    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test, inferences),

            #Our Learning Rate
            ScheduledHyperParamSetter('learning_rate', [(0, 1e-1), (41, 1e-2), (123, 1e-3), (200, 0.0002)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model(EPSILON, NUM_CLASS, NUM_UNITS),
        max_epoch = MAX_EPOCH,
    )

def eval_on_cifar(model_file):
    ds = get_data('test')
    print("Calling the Test Method")
    pred_config = PredictConfig(
        model=Model(n=NUM_UNITS),
        session_init=get_model_loader(model_file),
        input_names = ['input', 'label'],
        output_names = ['incorrect_vector']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    acc = RatioCounter()
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        acc.feed(o[0].sum(), batch_size)
    print("Error: {}".format(acc.ratio))

if __name__ == '__main__':
    assert tf.test.is_gpu_available(), "error3"
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=18)
    parser.add_argument('--load', help='load model')
    parser.add_argument('-e', '--epsilon', help='set epsilon')
    parser.add_argument('-o', '--output', help='output')
    feature_parser = parser.add_mutually_exclusive_group(required=True)
    feature_parser.add_argument('--cifar10', help='iscifar10', dest='dataset', action = 'store_true')
    feature_parser.add_argument('--cifar100', help='iscifar100', dest='dataset', action = 'store_false')

    parser.set_defaults(feature=True)

    args = parser.parse_args()
    NUM_UNITS = args.num_units
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print("os environ: {}", os.environ['CUDA_VISIBLE_DEVICES'])
        assert tf.test.is_gpu_available(), "errorr5"
    if args.epsilon:
        EPSILON = float(args.epsilon)
    if not args.dataset:
        print('args.dataset: {}'.format(args.dataset))
        IS_CIFAR10 = args.dataset
        NUM_CLASS = 100
    out_dir = ""
    if args.output:
        out_dir = "." + args.output
    print('epsilon = %f' % EPSILON)
    config = get_config(out_dir)
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))

    SyncMultiGPUTrainer(config).train()
