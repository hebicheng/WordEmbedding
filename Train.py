import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from WordEmbeddingDataset import WordEmbeddingDataset
from Model import EmbeddingModel

def Train():
    USE_CUDA = torch.cuda.is_available()
    random.seed(53113)
    np.random.seed(53113)
    torch.manual_seed(53113)
    if USE_CUDA:
        torch.cuda.manual_seed(53113)

    # setting hyper parameters
    NUM_EPOCHS = 2 # The number of epochs of training
    MAX_VOCAB_SIZE = 30000 # the vocabulary size
    BATCH_SIZE = 128 # the batch size
    LEARNING_RATE = 0.2 # the initial learning rate
    EMBEDDING_SIZE = 100

    # load data
    with open("data/text8.train.txt", "r") as fin:
        text = fin.read()

    text = text.split()

    vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
    vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
    idx_to_word = [word for word in vocab.keys()] 
    word_to_idx = {word:i for i, word in enumerate(idx_to_word)}

    word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3./4.)
    word_freqs = word_freqs / np.sum(word_freqs) # 用来做 negative sampling
    VOCAB_SIZE = len(idx_to_word)

    embedding_weights = np.load('embedding-100.npy')
    words = ['apple', 'man', 'train', 'dog', 'glove']
    for word in words:
        print(word)
        print(find_nearest(word,word_to_idx,embedding_weights, idx_to_word))
    # model = EmbeddingModel(MAX_VOCAB_SIZE , EMBEDDING_SIZE)
    # if USE_CUDA:
    #     model = model.cuda()

    # dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
    # dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # for e in range(NUM_EPOCHS):
    #     for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            
    #         input_labels = input_labels.long()
    #         pos_labels = pos_labels.long()
    #         neg_labels = neg_labels.long()
    #         if USE_CUDA:
    #             input_labels = input_labels.cuda()
    #             pos_labels = pos_labels.cuda()
    #             neg_labels = neg_labels.cuda()
                
    #         optimizer.zero_grad()
    #         loss = model(input_labels, pos_labels, neg_labels).mean()
    #         loss.backward()
    #         optimizer.step()

    #         if i % 100 == 0:
    #             print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))
                
    #     embedding_weights = model.input_embeddings()
    #     np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
    #     torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))

def find_nearest(word, word_to_idx,embedding_weights,idx_to_word):
    """
        用来查看词向量空间中与目标词最相似的词(夹角最小)
    """
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]