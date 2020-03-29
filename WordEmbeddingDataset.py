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


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, word_to_idx['<unk>']) for t in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
        
    def __len__(self):
        return len(self.text_encoded)
        
    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        '''
        K = 100 # number of negative samples
        C = 3 # nearby words threshold
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-C, idx)) + list(range(idx+1, idx+C+1))
        pos_indices = [i%len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices] 
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        
        return center_word, pos_words, neg_words