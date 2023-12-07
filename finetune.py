import os
import random
import numpy as np
import tensorflow as tf
from src import ModelOptions
from src import SimpsonsModel

'''
Run this with
python finetune.py \
    --checkpoints-path ./checkpoints/simpsons11 \
    --dataset-path ./dataset/simpsons_train_256 \
    --dataset simpsons \
    --batch-size 40 \
    --epochs 100 \
    --lr 5e-4 \
    --save-interval 80 \
    --lr-decay-rate 0.5 \
    --lr-decay-steps 50 \
    --validate 1 \
    --validate-interval 1 \
    --log 1 \
    --sample-interval 10

To test: 
python test.py \
  --checkpoints-path ./checkpoints/simpsons11 \
  --test-input ./dataset/test256 \
  --test-output ./output/test 
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
    # model.load()
    model.load_pretrained('./checkpoints/places365/')
    # model.learning_rate = tf.constant(1e-5)
    model.train()
    # model.test()