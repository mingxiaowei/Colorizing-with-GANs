import os
import random
import numpy as np
import tensorflow as tf
from src import ModelOptions
from src import Cifar10Model, Places365Model, SimpsonsModel

'''
Run this with
python finetune.py \
    --checkpoints-path ./checkpoints/simpsons \
    --dataset-path ./dataset/simpsons_train \
    --dataset simpsons \
    --batch-size 2 \
    --epochs 4 \
    --label-smoothing 1
'''
options = ModelOptions().parse()
tf.compat.v1.reset_default_graph()
tf.random.set_seed(options.seed)
np.random.seed(options.seed)
random.seed(options.seed)


# create a session environment
with tf.compat.v1.Session() as sess:

    model = SimpsonsModel(sess, options)

    if not os.path.exists(options.checkpoints_path):
        os.makedirs(options.checkpoints_path)

    if options.log:
        open(model.train_log_file, 'w').close()
        open(model.test_log_file, 'w').close()

    model.build()
    sess.run(tf.compat.v1.global_variables_initializer())
    model.load()
    model.train()
    # model.test()