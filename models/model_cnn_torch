"""
Created on 19/09/18.
Author: morgan
CNN for text classification in Pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNNClassifier(nn.Module):
    def __init__(self, batch_size, num_classes, in_channels, out_channels, kernel_heights,
                 stride, padding, keep_prob, vocab_size, embed_size, weights):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights,
        self.stride = stride
        self.padding = padding
        self.keep_prob = keep_prob
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.weights = weights

        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embed_size), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embed_size), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embed_size), stride, padding)
        self.dropout = nn.Dropout(keep_prob)
        self.fc = nn.Linear(len(kernel_heights)*out_channels, num_classes)

    def conv_block(self, input, conv_layer):
        # input: [batch_size, 1, seq_len, embed_size]
        # print('input size:', input.size())
        # input: [64, 1, 100, 300]
        conv_out = conv_layer(input) # [64, 128, 96, 1]
        # print('conv_out 0:', conv_out.size())
        conv_out = conv_out.squeeze(3) # [64, 128, 96]

        activation = F.relu(conv_out) # [64, 128, 96]
        # print('activation size 1:', )
        dim = activation.size()[-1] # last dimension of activiation, 96
        max_out = F.max_pool1d(activation, dim) # [batch_size, out_channels, 1] = [64, 128, 1]
        max_out = max_out.squeeze(2) # [batch_size, out_channels] =[64, 128]
        return max_out

    def forward(self, input_sentences, batch_size=None):
        # input: [64, 100]
        input = self.word_embeddings(input_sentences) # [batch_size, max_len, embed_size] [64, 100, 300]
        # print('input size2:', input.size())
        input = input.unsqueeze(1) # [batch_size, 1, max_len, embed_size] [64, 1, 100, 300]
        # print('input size3:', input.size())
        max_out1 = self.conv_block(input, self.conv1) # [batch_size, out_channels]
        max_out2 = self.conv_block(input, self.conv2) # [batch_size, out_channels] [64, 128]
        max_out3 = self.conv_block(input, self.conv3)# [batch_size, out_channels]

        all_out = torch.cat((max_out1, max_out2, max_out3), 1) # [64, 128*3]
        fc_in = self.dropout(all_out) # [64, 128*3]
        logits = self.fc(fc_in) # [64, 2]
        # print('logits;', logits.size())
        return logits


