"""
Created on 01/10/18.
Author: morgan
Copyright defined in text_classification/LICENSE.txt
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class RNNClassifier(nn.Module):
    def __init__(self, batch_size, num_classes, hidden_size, vocab_size, embed_size, weights):
        super(RNNClassifier, self).__init__()
        # weights: Pre-trained GloVe word_embeddings that we will use to create our word_embedding lookup table
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size) # initialize the lookup table
        # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers=2, bidirectional=True)
        self.proj = nn.Linear(4*hidden_size, num_classes)

    def forward(self, input_sentence):
        batch_size = input_sentence.size()[0]

        # input: [batch_size, seq_len], [64, 100]
        # print('input 0:', input_sentence.size())
        input = self.word_embeddings(input_sentence) # [batch_size, seq_len, embed_size]p
        # print('input 1:', input.size())
        input = input.permute(1, 0, 2).contiguous() # [seq_len, batch_size, embed_size]

        # Initiate hidden/cell state of the LSTM
        h_0 = Variable(torch.zeros(4, batch_size, self.hidden_size).cuda())
        # [4, batch_size, hidden_size]

        output, h_n = self.rnn(input, h_0)
        # h_n: [4, batch_size, hidden_size]
        # output: [max_len, batch_size, hidden]
        # print('h_n:', h_n.size())
        # print('output', output.size())
        h_n = h_n.permute(1, 0, 2).contiguous() #[batch_size, 4, hidden_size]
        # print('h_n1:', h_n.size())

        h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1]*h_n.size()[2])
        # [batch_size, 4*hidden_size]

        # print('h_n2:', h_n.size())
        # final_hidden_state: [1, batch_size, hidden_size]

        logtis = self.proj(h_n)
        # print('logtis:', logtis.size())
        # final_output: [batch_size, num_classes]

        return logtis
