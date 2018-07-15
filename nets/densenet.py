"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        #瓶颈层：控制maps
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        #block层
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5
    layers = 4


    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            pass
            ##########################
            # Put your code here.
            ##########################
            #[-1,320,320,3] to [-1,160,160,48]
            net = slim.conv2d(images, 2*growth, [7, 7], stride=2, scope='Conv_a')
            end_points['Conv_a']=net

            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_a')
            end_points['MaxPool_a'] = net

            net = block(net, layers, growth, scope='block1')
            end_points['block1'] = net

            net = slim.conv2d(net, reduce_dim(net), [1, 1], stride=1, scope='Compression_1')
            end_points['Compression_1'] = net

            net = slim.batch_norm(net, scope='trans1_bn')
            end_points['trans1_bn'] = net
            net = slim.conv2d(net, growth, [1, 1], stride=1, scope='trans1_conv')
            end_points['trans1_conv'] = net
            net = slim.avg_pool2d(net, [2, 2], stride=2, scope='trans1_avgPool')
            end_points['trans1_avgPool'] = net

            net = block(net, layers, growth, scope='block2')
            end_points['block2'] = net

            net = slim.conv2d(net, reduce_dim(net), [1, 1], stride=1, scope='Compression_2')
            end_points['Compression_2'] = net

            net = slim.batch_norm(net, scope='trans2_bn')
            end_points['trans2_bn'] = net
            net = slim.conv2d(net, growth, [1, 1], stride=1, scope='trans2_conv')
            end_points['trans2_conv'] = net
            net = slim.avg_pool2d(net, [2, 2], stride=2, scope='trans2_avgPool')
            end_points['trans2_avgPool'] = net

            net = block(net, layers, growth, scope='block3')
            end_points['block3'] = net

            net = slim.conv2d(net, reduce_dim(net), [1, 1], stride=1, scope='Compression_3')
            end_points['Compression_3'] = net

            net = slim.batch_norm(net, scope='trans3_bn')
            end_points['trans3_bn'] = net
            net = slim.conv2d(net, growth, [1, 1], stride=1, scope='trans3_conv')
            end_points['trans3_conv'] = net
            net = slim.avg_pool2d(net, [2, 2], stride=2, scope='trans3_avgPool')
            end_points['trans3_avgPool'] = net

            net = block(net, layers, growth, scope='block4')
            end_points['block4'] = net
            #全局平均池化
            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_average_pooling')
            end_points['global_average_pooling'] = net
            net = slim.flatten(net)
            end_points['Flatten'] = net
            #全连接
            net = slim.fully_connected(net, num_classes, scope='fc')
            end_points['fc'] = net
            #softmax
            logits =tf.nn.softmax(net, name='logits')
            end_points['logits'] = logits

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
