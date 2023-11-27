import os
import random
import numpy as np
import tensorflow as tf
from src import ModelOptions
from src import Cifar10Model, Places365Model, SimpsonsModel
from .dataset import CIFAR10_DATASET, PLACES365_DATASET

'''
Run this with
python finetune.py \
  --checkpoints-path /checkpoints/places364 \        # checkpoints path
  --test-input img/simpsons_small_test \         # test image(s) path
  --test-output ./checkpoints/output 
  --dataset simpsons
'''
options = ModelOptions().parse()
tf.reset_default_graph()

tf.set_random_seed(options.seed)
np.random.seed(options.seed)
random.seed(options.seed)


# create a session environment
with tf.Session() as sess:

    model = SimpsonsModel(sess, options)

    if not os.path.exists(options.checkpoints_path):
        os.makedirs(options.checkpoints_path)

    if options.log:
        open(model.train_log_file, 'w').close()
        open(model.test_log_file, 'w').close()

    model.build()
    sess.run(tf.global_variables_initializer())
    model.load()
    # model.train()
    model.test()