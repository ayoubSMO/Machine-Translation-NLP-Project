
import gensim
import nltk
from nltk.corpus import stopwords
import re
from Translator_bilingue import TranslatorBilingue
from gensim.models import Word2Vec
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
import numpy as np
import random



"""
# Created By : BOUREQBA Ayoub
# Github : Ayoub SMO
# LinkedIn : Ayoub-Boureqba

#:keyword this class is to convert all our data sentences to vector embedded
"""

class embedded_translation(TranslatorBilingue):

    """
    #:param
    #:return
    """
    def __init__(self ,path1 ,path2 ,lang1 ,lang2):
        self.biling_words = TranslatorBilingue.__init__(self,path1 ,path2 ,lang1,lang2)
        print("j'ai bien fait la construction ! ")

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

        return str(en_voc) ,str(fr_voc)


    def preprocessing(self ,first_voc ,sec_voc):

        # Removing the exrea caracters !
        first_voc = re.sub(r"[[0-9]*\]", " ", first_voc[:10000])
        sec_voc = re.sub(r"[[0-9]*\]", " ", sec_voc[:10000])

        first_voc = re.sub(r"\s+", " ", first_voc[:10000])
        sec_voc = re.sub(r"\s+", " ", sec_voc[:10000])

        first_voc = first_voc.lower()
        sec_voc = sec_voc.lower()
        print("j'ai supprimer extra chars")

        # tokenize and remove stop words !
        first_voc_tok_sent = nltk.sent_tokenize(first_voc)
        sec_voc_tok_sent = nltk.sent_tokenize(sec_voc)


        first_voc_tok_words = [nltk.word_tokenize(sentence) for sentence in first_voc_tok_sent]
        sec_voc_tok_words = [nltk.word_tokenize(sentence) for sentence in sec_voc_tok_sent]
        print("j'ai fait la tokenization")

        # Define lang1 and lang2 stop words

        stopwords_lan1 = stopwords.words(str(self.lang1))
        stopwords_lan2 = stopwords.words(str(self.lang2))

        # remove the stop words from the texts

        for i ,_ in enumerate(first_voc_tok_words):
            first_voc_tok_words[i] = [word for word in first_voc_tok_words[i] if word not in stopwords_lan1]

        for j, x in enumerate(sec_voc_tok_words):
            sec_voc_tok_words[i] = [word_ for word_ in sec_voc_tok_words[i] if word_ not in stopwords_lan2]

        print("j'ai supprimer les stopwords!")



        return first_voc_tok_sent,first_voc_tok_words ,sec_voc_tok_sent ,sec_voc_tok_words



    def word2vect(self ,tok_Words):
        print("J'ai bien construit le model")

        return Word2Vec(sentences = tok_Words,
                                size = 400,
                                window = 10,
                                iter = 20,)

    def get_closest_english_words(sefl ,sec_word,first_voc ,sec_voc, k ,W ,X):

        fr_obj = sec_voc.wv.get_vector(sec_word)
        #print("----" ,fr_obj)
        aligne_fr = W.dot(fr_obj.T)
        en_embeds = X[:,1]
        norm_prod = np.linalg.norm(aligne_fr) * np.linalg.norm(en_embeds, axis=0)
        #print(aligne_fr ,'\n',norm_prod)
        #aligne_fr = aligne_fr.reshape(aligne_fr.shape[0], 6)
        #norm_prod = norm_prod.reshape(norm_prod.shape[0], 6)
        #print(en_embeds)
        print(en_embeds.shape, '\n', norm_prod.shape)
        scores =((en_embeds).dot(aligne_fr)) / norm_prod

        #scores.reshape(scores.shape[0] ,1)
        best_k = np.flip(np.argsort(scores))[:k]
        print(best_k)
        return ([first_voc.wv.index2word[best_k]])





