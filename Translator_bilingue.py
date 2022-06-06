"""
# Created By : BOUREQBA Ayoub
# Github : Ayoub SMO
# LinkedIn : Ayoub-Boureqba
"""


import gzip
import io

import tqdm
import os
import random
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GRU ,Input,Dense,TimeDistributed,SimpleRNN,Dropout,Reshape, Activation
from keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

class TranslatorBilingue :

    # Constructeur de notre machine de traduction
    # en deffinissent les fichiers des deux langues
    def __init__(self ,pathLang1 ,pathLang2 ,lang1 ,lang2):
        self.pathLang1 = pathLang1
        self.pathLang2 = pathLang2
        self.lang1 = lang1
        self.lang2 = lang2
        print("___ j'ai bien fait la construction !")

    def set_embedded_lang1(self, pathLang1):
        self.pathLang1 = pathLang1

    def set_embedded_lang2(self, pathLang2):
        self.pathLang2 = pathLang2



    def load_data_for_new_vocab(self):
        try :

            with open(self.pathLang1 ,'rt',encoding="utf8",newline='\n',errors='ignore') as f:
                en_voc = f.readlines()

        except Exception:
            print("you have problemes in file 1 !")

        try :
            with open(self.pathLang2 ,'rt',encoding="utf8",newline='\n',errors='ignore') as f:
                fr_voc = f.readlines()
        except Exception:
            print("you have problemes in file 2 !")

        print("loading data is finish !")

        return en_voc ,fr_voc


    # Data Prepocessing

    def tokenize(self ,sent):
        """
            Tokenize sentence
            :param sent: List of sentences
            :return: Tuple of (tokenized sent data, tokenizer used to tokenize sent)
        """
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sent)

        return tokenizer.texts_to_sequences(sent) ,tokenizer


    # When batching the sequence of word ids together, each sequence needs to be the same length. Since sentences are dynamic in length, we can add padding to the end of the sequences to make them the same length.
    def pad(self,seq, length=None):

        sequences = pad_sequences(seq, maxlen=length, padding='post')

        return sequences

    def preprocess(self,x, y):
        """
        Preprocess x and y
        :param x: Feature  Listof sentences
        :param y: Label List of sentences
        :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
        """
        preprocess_x, x_tk = self.tokenize(x)
        preprocess_y, y_tk = self.tokenize(y)

        preprocess_x = self.pad(preprocess_x)
        preprocess_y = self.pad(preprocess_y)

        # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
        preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

        return preprocess_x, preprocess_y, x_tk, y_tk

    def logits_to_text_(self,logits, tokenizer):
        """
        Turn logits from a neural network into text using the tokenizer
        :param logits: Logits from a neural network
        :param tokenizer: Keras Tokenizer fit on the labels
        :return: String that represents the text of the logits
        """
        index_to_words = {id: word for word, id in tokenizer.word_index.items()}
        index_to_words[0] = '<PAD>'

        return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

    def logits_to_text(self, logits, tokenizer):
        index_to_words = {id: word for word, id in tokenizer.word_index.items()}
        index_to_words[0] = '&amp;lt;PAD&amp;gt;'

        return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

    # Simple Recur Neural Network model

    def simple_RNN(self ,input_shape ,output_sequence_length ,lang1_size ,lang2_size):
        model = Sequential()
        model.add(SimpleRNN(units = output_sequence_length,
                            activation='tanh',
                            return_sequences=True,
                            input_shape=input_shape[1:]))
        model.add(TimeDistributed(Dense(2 * lang1_size ,activation='relu')))
        model.add(TimeDistributed(Dense(lang1_size ,activation='softmax')))
        learning_rate = 0.001

        model.compile(loss = 'categorical_crossentropy',
                      optimizer=Adam(learning_rate),
                      metrics=['accuracy'])
        return model

    def simple_RNN_(self ,input_shape, output_sequence_length, lang1_size, lang2_size):
        learning_rate = 1e-3
        input_seq = Input(input_shape[1:])
        rnn = GRU(64, return_sequences=True)(input_seq)
        logits = TimeDistributed(Dense(lang1_size))(rnn)
        model = Model(input_seq, Activation('softmax')(logits))
        model.compile(loss=sparse_categorical_crossentropy,
                      optimizer=Adam(learning_rate),
                      metrics=['accuracy'])

        return model




