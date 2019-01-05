#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 Wan Li. All Rights Reserved
#
########################################################################

"""
File: tsne_tensorflow.py
Author: Wan Li
Date: 2019/01/05 21:52:03
"""

import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

def tsne_project(X, y, logdir="./tflog"):
    """
        tSNE project
    """
    label_fn = "labels.csv"
    embeddings = tf.Variable(X, trainable=False, name='embeddings')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(logdir, sess.graph)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embeddings.name
        embedding.metadata_path = os.path.basename(label_fn)
        projector.visualize_embeddings(writer, config)

        saver_embed = tf.train.Saver([embeddings])
        saver_embed.save(sess, '{}/embeddings_viz.ckpt'.format(logdir), 1)

    # save metadata
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    with open("{}/{}".format(logdir, label_fn), "w") as fd:
        for lbl in y:
            fd.write("{}\n".format(str(lbl)))
    print("Try: tensorboard --logdir {} --port 8008".format(logdir))

if __name__ == "__main__":
    from sklearn import datasets
    digits = datasets.load_digits()
    X = digits.data[:500]
    y = digits.target[:500]
    tsne_project(X, y)