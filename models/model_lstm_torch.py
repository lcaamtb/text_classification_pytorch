"""
Created on 01/10/18.
Author: morgan
Copyright defined in text_classification/LICENSE.txt
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, num_classes, hidden_size, vocab_size, embed_size, weights):
        super(LSTMClassifier, self).__init__()
        # weights: Pre-trained GloVe word_embeddings that we will use to create our word_embedding lookup table
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size) # initialize the lookup table
        # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.proj = nn.Linear(hidden_size, num_classes)

    def forward(self, input_sentence):
        batch_size = input_sentence.size()[0]

        # input: [batch_size, seq_len], [64, 100]
        # print('input 0:', input_sentence.size())
        input = self.word_embeddings(input_sentence) # [batch_size, seq_len, embed_size]p
        # print('input 1:', input.size())
        input = input.permute(1, 0, 2) # [seq_len, batch_size, embed_size]
        # print('input 2:', input.size())

        # Initiate hidden/cell state of the LSTM

        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        # print('h0:', h_0.size())
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        # print('c0:', c_0.size())

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        # output: [max_len, batch_size, hidden_size]
        # final_hidden/cell_state: [1, batch_size, hidden_size]
        # print('output:', output.size())
        # print('final_hidden_state', final_hidden_state.size())
        # print('final_cell_state', final_cell_state.size())

        # final_hidden_state: [1, batch_size, hidden_size]
        final_output = self.proj(final_hidden_state[-1])
        # final_output: [batch_size, num_classes]
        # print('final_output size:', final_output.size())
        return final_output