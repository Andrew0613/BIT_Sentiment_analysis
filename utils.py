import re
import string
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import jieba
stopword = [line.strip() for line in open('data/stopwords/stopword.txt', 'r',encoding='utf-8',errors='ignore').readlines()]
def get_content(path):
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        content=''
        for l in f:
            l=l.strip().replace(u'\u3000',u'')
            content+=l
    return content        
def get_file_content(path):
    flist=os.listdir(path)
    flist=[os.path.join(path,x) for x in flist]
    corpus=[get_content(x) for x in flist]
    return corpus
def process_sentence(sentence):
    stemmer = PorterStemmer()
    sentence = re.sub(r'\$\w*', '', sentence)
    # remove old style retweet text "RT"
    sentence = re.sub(r'^RT[\s]+', '', sentence)
    # remove hyperlinks
    sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence)
    # remove hashtags
    # only removing the hash # sign from the word
    sentence = re.sub(r'#', '', sentence)
    sentence_tokens = jieba.lcut(sentence=sentence)
    sentence_clean =[]
    #clean data 
    for word in sentence_tokens:
        if (word not in stopword and  # remove stopwords
                word not in string.punctuation and word != ' '):  # remove punctuation
            # tweets_clean.append(word)
            # stem_word = stemmer.stem(word)  # stemming word
            sentence_clean.append(word)

    return sentence_clean

def build_freqs(sentences, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, sentence in zip(yslist, sentences):
        for word in process_sentence(sentence):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs
def lookup(freqs, word, label):
    '''
    Input:
        freqs: a dictionary with the frequency of each pair (or tuple)
        word: the word to look up
        label: the label corresponding to the word
    Output:
        n: the number of times the word with its corresponding label appears.
    '''
    n = 0  # freqs.get((word, label), 0)

    pair = (word, label)
    if (pair in freqs):
        n = freqs[pair]

    return n