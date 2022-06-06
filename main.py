"""
# Created By : BOUREQBA Ayoub
# Github : Ayoub SMO
# LinkedIn : Ayoub-Boureqba
"""

from Translator_bilingue import TranslatorBilingue
from embedded_translation import embedded_translation
import numpy as np
import gensim.models.keyedvectors as w2v



if __name__== "__main__":

    #machine = TranslatorBilingue("europarl-v7.fr-en.en","europarl-v7.fr-en.fr","english" ,"frensh")

    #data1 = machine.load_vectors()

    #english ,frensh = machine.load_data_for_new_vocab()

    """
    for i in range(3):
        print("all data english is in  {} : {}".format(i + 1 ,frensh[i]))
        print("all data frensh is in {} : {}".format(i + 1 ,english[i]))


    preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
        machine.preprocess(english[:100], frensh[:100])

    tmp_x = machine.pad(preproc_english_sentences, preproc_french_sentences.shape[1])
    tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

    model = machine.simple_RNN_(tmp_x.shape ,preproc_french_sentences.shape[1],len(english_tokenizer.word_index),len(french_tokenizer.word_index))

    model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

    print(machine.logits_to_text(model.predict(tmp_x[:1])[0], french_tokenizer))
    """

    machine = embedded_translation("europarl-v7.fr-en.en","europarl-v7.fr-en.fr","english" ,"french")
    first_voc, sec_voc = machine.load_data_for_new_vocab()

    # NLP - Station - Africa - Fracture Africha - Yousfi - Benhlima -

    first_voc_tok_sent,first_voc_tok_words ,sec_voc_tok_sent ,sec_voc_tok_words = machine.preprocessing(first_voc ,sec_voc)
    #print(first_voc_tok_words)
    word2vec_lang1 = machine.word2vect(first_voc_tok_words)
    word2vec_lang2 = machine.word2vect(sec_voc_tok_words)

    #word2vec_lang1.wv.save('word2vec_lang1.bin')
    #word2vec_lang2.wv.save('word2vec_lang2.bin')

    mots_transparent = [word for word in word2vec_lang1.wv.vocab if word in word2vec_lang2.wv.vocab]
    print(mots_transparent)

    X, Y = np.empty([400, len(mots_transparent)]), np.empty([400, len(mots_transparent)])

    for i, word in enumerate(mots_transparent):
        X[:, i] = word2vec_lang1.wv.get_vector(word)
        Y[:, i] = word2vec_lang2.wv.get_vector(word)
    assert X.shape[0] == 400 and Y.shape[0] == 400

    U, sigma, Vtranspose = np.linalg.svd(Y.dot(X.T))
    W = U.dot(Vtranspose)

    #print(word2vec_lang2.wv.vocab)
    print(machine.get_closest_english_words("parlement" ,word2vec_lang1 ,word2vec_lang2 ,1 ,W ,X))




