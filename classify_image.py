# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = None

class NodeLookup(object):
  def __init__(self, label_lookup_path=None):
    self.node_lookup = self.load(label_lookup_path)

  def load(self, label_lookup_path):
    node_id_to_name = {}
    f = open(label_lookup_path, encoding="utf-8")
    line = f.readline()
    while line:
        line = line.strip('\n')
        line_info = line.split(':')
        line_id = int(line_info[0])
        line_name = line_info[1]
        node_id_to_name[line_id] = line_name
        line = f.readline()
    f.close()
    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  model_path_='my_inception_v4_freeze.pb'
  with tf.gfile.FastGFile(model_path_, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  #with tf.Graph().as_default():
    #image_data = tf.gfile.FastGFile(image, 'rb').read()
    #image_data = tf.image.decode_jpeg(image_data)
    #image_data = preprocess_for_eval(image_data, 299, 299)
    #image_data = tf.expand_dims(image_data, 0)
    #with tf.Session() as sess:
      #image_data = sess.run(image_data)
  image_data = open(image, 'rb').read()
  print(type(image_data))
  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('prediction:0')
    predictions = sess.run(softmax_tensor, {'input:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    label_path_= 'labels.txt'
    node_lookup = NodeLookup(label_path_)
    num_top_predictions_=1
    top_k = predictions.argsort()[-num_top_predictions_:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%d %s (score = %.5f)' % (node_id, human_string, score))


def main(_):
  image = 'test.jpeg'
  run_inference_on_image(image)


if __name__ == '__main__':
  tf.app.run()