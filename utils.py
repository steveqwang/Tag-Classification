# -*- coding: utf-8 -*-

import operator
import numpy as np
import tensorflow as tf


class Vocab:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}

        self.word_count = {}

        self.size = 8000

    def hasWord(self, word):
        return word in self.word2id.keys()

    def build(self, filenames):

        spec = '`~!@#$%^&*()_+-=[]\{}|;:,./<>?[]{}·”…□○●、。《》「」『』〖〗'

        for filename in filenames:
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    for s in spec:
                        line = line.replace(s, ' ')
                    words = line.split()

                    for word in words:
                        word = word.lower()
                        if word not in self.word_count:
                            self.word_count[word] = 1
                        else:
                            self.word_count[word] += 1

        sorted_dic = sorted(self.word_count.items(), key=operator.itemgetter(1), reverse=True)[:self.size-1]

        self.word2id['<unknown>'] = 1
        self.id2word[1] = '<unknown>'

        index = 2
        for key, _ in sorted_dic:
            self.word2id[key] = index
            self.id2word[index] = key
            index += 1


def preprocess(filename, vocab):

    spec = '`~!@#$%^&*()_+-=[]\{}|;:,./<>?[]{}·”…□○●、。《》「」『』〖〗'

    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words_list = []
            # word_s = ""
            for s in spec:
                line = line.replace(s, ' ')
            words = line.split()

            for word in words:
                word = word.lower()
                if not vocab.hasWord(word):
                    word = '<unknown>'
                # word_s = word_s + ' ' + word
                words_list.append(vocab.word2id[word])
            res.append(words_list)
            # res.append(word_s)
    return res


class DataLoader:

    def __init__(self, X, Y, seq_length):

        self.seq_length = seq_length

        self.X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.seq_length, padding='post')
        self.Y = np.array(Y)
