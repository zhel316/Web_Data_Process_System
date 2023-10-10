import random
import numpy as np
import os
import sys
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import torch
from torch.autograd import Variable

import RE_model.utils as utils


class DataLoader(object):
    def __init__(self, data_dir, embedding_file, word_emb_dim, max_len=100, pos_dis_limit=50,
                 pad_word='<pad>', unk_word='<unk>', other_label='Other', gpu=0):
        self.data_dir = data_dir  # data root directory
        self.embedding_file = embedding_file  # embedding file path
        self.max_len = max_len  # specified max length of the sentence
        self.word_emb_dim = word_emb_dim
        self.limit = pos_dis_limit  # limit of relative distance between word and entity
        self.pad_word = pad_word  # padding word
        self.unk_word = unk_word  # unknown word。
        self.other_label = other_label  # the label of other relations that do not consider to be evaluated
        self.gpu = gpu

        self.word2idx = dict()
        self.label2idx = dict()

        self.embedding_vectors = list()  # the index is consistent with word2idx
        self.unique_words = list()  # unique words of datasets

        self.original_words_num = 0
        self.lowercase_words_num = 0
        self.zero_digits_replaced_num = 0
        self.zero_digits_replaced_lowercase_num = 0
        
        if pad_word is not None:
            self.pad_idx = len(self.word2idx)  # default: 0
            self.word2idx[pad_word] = self.pad_idx
            self.embedding_vectors.append(utils.generate_zero_vector(self.word_emb_dim))

        if unk_word is not None:
            self.unk_idx = len(self.word2idx)  # default: 1
            self.word2idx[unk_word] = self.unk_idx
            self.embedding_vectors.append(utils.generate_random_vector(self.word_emb_dim))
        
        # load unique words
        vocab_path = os.path.join(self.data_dir, 'words.txt')
        with open(vocab_path, 'r') as f:
            for line in f:
                self.unique_words.append(line.strip())
        
        # load labels (labels to indices)
        labels_path = os.path.join(data_dir, 'labels.txt')
        with open(labels_path, 'r') as f:
            for i, line in enumerate(f):
                self.label2idx[line.strip()] = i

        # get the relation labels to be evaluated
        other_label_idx = self.label2idx[self.other_label]
        self.metric_labels = list(self.label2idx.values())
        self.metric_labels.remove(other_label_idx)

    # return tensor.cuda(device=self.gpu)
    # device = torch.device('cuda')
    # return tensor.to(device)
    def tensor_ensure_gpu(self, tensor):
        """Shift tensors to GPU if available"""
        if self.gpu >= 0:
            return tensor.cuda(device=self.gpu)
        else:
            return tensor

    def get_loaded_embedding_vectors(self):
        """Get word embedding vectors"""
        return torch.FloatTensor(np.asarray(self.embedding_vectors))

    def load_embeddings_from_file_and_unique_words(self, emb_path, emb_delimiter=' ', verbose=True):
        embedding_words = [emb_word for emb_word, _ in self.load_embeddings_from_file(emb_path=self.embedding_file, 
                                                                                      emb_delimiter=emb_delimiter)]
        emb_word2unique_word = dict()  # emb_word <-> [unique_word_1, unique_word_2, ...]
        out_of_vocab_words = list()  # not in the pretrained vocabulary
        for unique_word in self.unique_words:
            emb_word = self.get_embedding_word(unique_word, embedding_words)
            if emb_word is None: # Indicates that the various forms of the word are not in the pre-trained vocabulary
                out_of_vocab_words.append(unique_word)
            else:
                if emb_word not in emb_word2unique_word:
                    emb_word2unique_word[emb_word] = [unique_word] # List form, convenient for the following append()
                else:
                    emb_word2unique_word[emb_word].append(unique_word) # emb_word is a word that ignores case
                    # here is to put the various forms of the word under a unified key

        for emb_word, emb_vector in self.load_embeddings_from_file(emb_path=self.embedding_file,
                                                                   emb_delimiter=emb_delimiter):
            if emb_word in emb_word2unique_word:
                for unique_word in emb_word2unique_word[emb_word]:
                    self.word2idx[unique_word] = len(self.word2idx)
                    self.embedding_vectors.append(emb_vector) # 只有大小写区别的词使用同一词向量

    def load_embeddings_from_file(self, emb_path, emb_delimiter):
        """Load word embedding from file"""
        with open(emb_path, 'r') as f:
            for line in f:
                values = line.strip().split(emb_delimiter)
                word = values[0]
                # map(fun,x)
                # filter filters blank characters, and then map converts str type to Wei float type
                emb_vector = list(map(lambda emb: float(emb),
                                  filter(lambda val: val and not val.isspace(), values[1:])))
                yield word, emb_vector

    def get_embedding_word(self, word, embedding_words):
        """Mapping of words in datsets into embedding words"""
        if word in embedding_words:
            self.original_words_num += 1
            return word
        elif word.lower() in embedding_words:
            self.lowercase_words_num += 1
            return word.lower()
        elif re.sub(r'\d', '0', word) in embedding_words:
            self.zero_digits_replaced_num += 1
            return re.sub(r'\d', '0', word)
        elif re.sub(r'\d', '0', word.lower()) in embedding_words:
            self.zero_digits_replaced_lowercase_num += 1
            return re.sub(r'\d', '0', word.lower())
        return None

    def load_sentences_labels(self, sentences_file, labels_file, d):
        """Loads sentences and labels from their corresponding files. 
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sents, pos1s, pos2s = list(), list(), list()
        labels = list()

        # Replace each token by its index if it is in vocab, else use index of unk_word
        with open(sentences_file, 'r') as f:
            for i, line in enumerate(f):
                e1, e2, sent = line.strip().split('\t')
                words = sent.split(' ')
                e1_start = e1.split(' ')[0] if ' ' in e1 else e1
                e2_start = e2.split(' ')[0] if ' ' in e2 else e2
                e1_idx = words.index(e1_start)  # e1 index in the words/sent
                e2_idx = words.index(e2_start)  # e2 index in the words/sent
                sent, pos1, pos2 = list(), list(), list()
                for idx, word in enumerate(words):
                    emb_word = self.get_embedding_word(word, self.word2idx)
                    if emb_word:
                        sent.append(self.word2idx[word])
                    else:
                        sent.append(self.unk_idx)
                    pos1.append(self.get_pos_feature(idx - e1_idx))
                    pos2.append(self.get_pos_feature(idx - e2_idx))
                sents.append(sent)
                pos1s.append(pos1)
                pos2s.append(pos2)
        
        # Replace each label by its index
        with open(labels_file, 'r') as f:
            for line in f:
                idx = self.label2idx[line.strip()]
                labels.append(idx)

        # Check to ensure there is a tag for each sentence
        assert len(labels) == len(sents)

        # Storing data and labels in dict d
        d['data'] = {'sents': sents, 'pos1s': pos1s, 'pos2s': pos2s}
        d['labels'] = labels
        d['size'] = len(sents)

    def get_pos_feature(self, x):
        """Clip the relative postion range:
            -limit ~ limit => 0 ~ limit * 2+2
        """
        if x < -self.limit:
            return 0
        elif x >= -self.limit and x <= self.limit:
            return x + self.limit + 1
        else:
            return self.limit * 2 + 2

    def load_pre(self,sentence):
        '''Loads the data and prepocesses for net

        :Args:
            sentence: signal sentence

        :return:
            tensors for CNN
        '''
        data = dict()

        e1, e2, sent = sentence.strip().split('\t')
        # words = sent.split(' ')
        words = word_tokenize(sent)
        e1_start = e1.split(' ')[0] if ' ' in e1 else e1
        e2_start = e2.split(' ')[0] if ' ' in e2 else e2
        e1_idx = words.index(e1_start)  # e1 index in the words/sent
        e2_idx = words.index(e2_start)  # e2 index in the words/sent
        sent, pos1, pos2 = list(), list(), list()
        for idx, word in enumerate(words):
            emb_word = self.get_embedding_word(word, self.word2idx)
            #print(word,emb_word)
            if emb_word:
                sent.append(self.word2idx[emb_word])
            else:
                sent.append(self.unk_idx)
            pos1.append(self.get_pos_feature(idx - e1_idx))
            pos2.append(self.get_pos_feature(idx - e2_idx))
        data['data'] = {'sents': [sent], 'pos1s': [pos1], 'pos2s': [pos2]}
        # data['labels'] = labels
        data['size'] = 1

        batch_sents = data['data']['sents']
        batch_pos1s = data['data']['pos1s']
        batch_pos2s = data['data']['pos2s']

        batch_max_len = self.max_len
        batch_size = 1
        batch_data_sents = self.pad_idx * np.ones((batch_size, batch_max_len))
        batch_data_pos1s = (self.limit * 2 + 2) * np.ones((batch_size, batch_max_len))
        batch_data_pos2s = (self.limit * 2 + 2) * np.ones((batch_size, batch_max_len))

        j = 0
        cur_len = len(batch_sents[j])
        min_len = min(cur_len, batch_max_len)
        batch_data_sents[j][:min_len] = batch_sents[j][:min_len]
        batch_data_pos1s[j][:min_len] = batch_pos1s[j][:min_len]
        batch_data_pos2s[j][:min_len] = batch_pos2s[j][:min_len]

        # Convert indices data to torch LongTensors, and shift tensors to GPU if available.
        batch_data_sents = self.tensor_ensure_gpu(torch.LongTensor(batch_data_sents))
        batch_data_pos1s = self.tensor_ensure_gpu(torch.LongTensor(batch_data_pos1s))
        batch_data_pos2s = self.tensor_ensure_gpu(torch.LongTensor(batch_data_pos2s))

        batch_data = {'sents': batch_data_sents, 'pos1s': batch_data_pos1s, 'pos2s': batch_data_pos2s}
        return batch_data



    def load_data(self, data_type):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        data = dict()
        
        if data_type in ['train', 'val', 'test']:
            sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
            labels_file = os.path.join(self.data_dir, data_type, 'labels.txt')
            self.load_sentences_labels(sentences_file, labels_file, data)
        else:
            raise ValueError("data type not in ['train', 'val', 'test']")
        return data

    def data_iterator(self, data, batch_size, shuffle='False'):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'labels' and 'size'
            batch_size: (int) batch size
            shuffle: (bool) whether the data should be shuffled
            
        Yields:
            batch_data: (tensor) shape: batch_size x max_seq_len
            batch_tags: (tensor) shape: batch_size x max_seq_len
        """
        order = list(range(data['size']))
        if shuffle:
            random.seed(230) # 保证结果可以复现
            random.shuffle(order)
        # one pass over data
        for i in range((data['size'])//batch_size):
            # fetch data and labels
            batch_sents = [data['data']['sents'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_pos1s = [data['data']['pos1s'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_pos2s = [data['data']['pos2s'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_labels = [data['labels'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]

            # Compute length of longest sentence in batch, and given batch_max_len.
            """每个batch填充的最大长度不一样"""
            # temp_len = max([len(s) for s in batch_sents])
            # batch_max_len = temp_len if temp_len < self.max_len else self.max_len
            batch_max_len = self.max_len

            """分为词特征和位置特征，词特征填充pad，位置特征填self.limit*2+2"""
            batch_data_sents = self.pad_idx * np.ones((batch_size, batch_max_len))
            batch_data_pos1s = (self.limit * 2 + 2) * np.ones((batch_size, batch_max_len))
            batch_data_pos2s = (self.limit * 2 + 2) * np.ones((batch_size, batch_max_len))
            for j in range(batch_size):
                cur_len = len(batch_sents[j])
                min_len = min(cur_len, batch_max_len)
                batch_data_sents[j][:min_len] = batch_sents[j][:min_len]
                batch_data_pos1s[j][:min_len] = batch_pos1s[j][:min_len]
                batch_data_pos2s[j][:min_len] = batch_pos2s[j][:min_len]

            # Convert indices data to torch LongTensors, and shift tensors to GPU if available.
            batch_data_sents = self.tensor_ensure_gpu(torch.LongTensor(batch_data_sents))
            batch_data_pos1s = self.tensor_ensure_gpu(torch.LongTensor(batch_data_pos1s))
            batch_data_pos2s = self.tensor_ensure_gpu(torch.LongTensor(batch_data_pos2s))
            batch_labels = self.tensor_ensure_gpu(torch.LongTensor(batch_labels))

            batch_data = {'sents': batch_data_sents, 'pos1s': batch_data_pos1s, 'pos2s': batch_data_pos2s}
            yield batch_data, batch_labels

