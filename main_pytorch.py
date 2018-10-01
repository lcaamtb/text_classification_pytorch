"""
Created on 19/09/18.
Author: morgan
Copyright defined in text_classification/LICENSE.txt
"""
import os

from data.build_vocab_pytorch import load_mr_dataset
from models.model_cnn_torch import CNNClassifier
from models.model_lstm_torch import LSTMClassifier
from models.model_rnn_torch import RNNClassifier
from hparams import HyperParameters as hp
import torch
import torch.nn.functional as F


def clip_gradient(model, clip_value):
    # gradients are clipped in the range [-clip_value, clip_value]
    # params with not None gradients
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp.learning_rate)
    steps = 0
    model.train()

    for idx, batch in enumerate(train_iter):
        # print('batch:')
        # print(batch.text.size())
        # print(batch.label.size())

        text = batch.text # [batch_size, max_len] =64, 100

        target = batch.label

        # print('type target:', type(target))
        target = target.long() # conver from tensor.FloatTensor to LongTensor

        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()

        prediction = model.forward(text) # [batch_size, num_classes]
        loss = F.cross_entropy(prediction, target) #??????/
        # print('loss: ', loss)

        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        # print('num_correct', num_corrects)
        acc = 100.0 * num_corrects/len(batch)
        # print('acc', acc)
        # print(type(loss)) # float
        # backward
        optimizer.zero_grad() # since backward() accumulates gradients, we dont wanna mix up gradients between mini batches, have to zero gradients out at the start of a new minibatch.
        loss.backward()
        clip_gradient(model, 1e-1) #gradients clipped in the range [-1e-1, 1e-1]
        optimizer.step() # update weights
        # print('loss type', type(loss))
        steps += 1
        if steps % 100 == 0:
            print('epoch:', epoch+1, 'train loss:', loss, 'train accu:', acc)
        total_epoch_acc += acc
        total_epoch_loss += loss
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, dev_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    for idx, batch in enumerate(dev_iter):
        text = batch.text
        target = batch.label
        target = target.long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        prediction = model.forward(text)
        loss = F.cross_entropy(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects / len(batch)
        total_epoch_acc += acc
        total_epoch_loss += loss
    return total_epoch_loss/len(dev_iter), total_epoch_acc/len(dev_iter)


path = os.getcwd()
path = os.path.dirname(path)
pos_path = os.path.join(path, "data/positive.txt")
neg_path = os.path.join(path, "data/negative.txt")
save_path = os.path.join(path, 'result/lstm_model.pt')
# print('pos path :', pos_path)
train_iter, dev_iter, TEXT, word_embeddings = load_mr_dataset(pos_path, neg_path)
# print('word_embed size:', word_embeddings.size())
# print(TEXT.vocab.stoi)
vocab_size = len(TEXT.vocab)
model = RNNClassifier(hp.batch_size, hp.num_classes, hp.hidden_size, vocab_size, hp.embed_size, word_embeddings)
# model = model.cuda()

for epoch in range(hp.num_epochs):
    train_loss, train_acc = train_model(model, train_iter, epoch)

torch.save(model, save_path) # save the whole model
print('train_loss ', train_loss, 'train acc=', train_acc)
model_new = torch.load(save_path)
print(model_new)
eval_loss, eval_acc = eval_model(model_new, dev_iter)
print('eval_loss ', eval_loss, 'eval acc=', eval_acc)
