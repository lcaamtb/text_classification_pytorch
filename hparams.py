"""
Created on 19/09/18.
Author: morgan
Copyright defined in text_classification/LICENSE.txt
"""

class HyperParameters:
    # model params
    embed_size = 300
    num_filters = 128
    num_classes = 2
    dropout_keep_prob = 0.1
    log_dir = "log_dir"
    max_len = 100

    stride = 1
    padding = 0
    keep_prob = 0.7

    filter_sizes = (3, 4, 5)

    # train params

    batch_size = 32
    hidden_size = 64
    learning_rate = 0.001
    num_epochs = 50
    evaluate_every = 100
    checkpoint_every = 100
    num_checkpoints = 5
    l2_reg_lambda = 0.01



