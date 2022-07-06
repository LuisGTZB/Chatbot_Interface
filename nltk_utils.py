import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Separacion de las palabras
    en tokes:
    sentence = [Hola como estas?]
     -> ["hola", "como", "Estas", "?"]
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = Encuentra la raiz de la palabra
    words = ["organizar", "organizando", "organizas"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

#algoritmo de la bolsa o cubeta
def bag_of_words(tokenized_sentence, words):
    
    #se procesan las palabras
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag