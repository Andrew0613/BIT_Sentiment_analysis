import numpy as np
from utils import *
def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0
    p_w_pos_sum = 0
    p_w_neg_sum = 0
    #单词表中单词的个数
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    #postive词的个数以及negative词的个数
    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]
    #文档数量
    D = len(train_y)
    # positive文档数量
    D_pos = sum(train_y)
    # negative文档数量
    D_neg = (D - D_pos)
    # 对数先验
    logprior = np.log(D_pos) - np.log(D_neg)
    for word in vocab:
        # 查询该词的为positive的频率和negative的频率
        freq_pos = lookup(freqs,word,1)
        freq_neg = lookup(freqs,word,0)
        # 计算概率
        p_w_pos = (freq_pos + 1)/(N_pos+V)
        p_w_neg = (freq_neg + 1)/(N_neg+V)
        
        p_w_pos_sum += p_w_pos
        p_w_neg_sum += p_w_neg
        #计算似然
        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)       
    return logprior, loglikelihood
def naive_bayes_predict(sentence, logprior, loglikelihood):
    #对于句子进行处理：分词加清洗
    word_l = process_sentence(sentence)
    # 初始化概率
    p = 0
    p += logprior
    #计算似然
    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood.get(word)
    return p
def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    accuracy = 0  
    total_loss = 0
    y_hats = []
    for sentence in test_x:
        if naive_bayes_predict(sentence, logprior, loglikelihood) > 0:
            y_hats.append(1)
        else:
            y_hats.append(0)
    y_hats = np.asarray(y_hats)
    test_y = np.squeeze(test_y)
    count =0
    for i in range(len(test_y)):
        if (test_y[i] == y_hats[i]):
            count = count+ 1
        else:
            count
    accuracy = count/(len(test_y))
    return accuracy
    # return accuracy

pos_comment=get_file_content('data/10000/pos')
neg_comment=get_file_content('data/10000/neg')
test_pos = pos_comment[4000:]
train_pos = pos_comment[:4000]
test_neg = neg_comment[4000:]
train_neg = neg_comment[:4000]
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)
train_x = train_pos + train_neg 
test_x = test_pos + test_neg
freqs = build_freqs(train_x, train_y)
logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
print("Naive Bayes accuracy = %0.4f" %
      (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

tfidf_vectorizer=TfidfVectorizer(min_df=1,norm='l2',smooth_idf=True,use_idf=True,ngram_range=(1,1))
train_features= tfidf_vectorizer.fit_transform(train_x)
test_features= tfidf_vectorizer.transform(test_x)

classifier=MultinomialNB()
classifier.fit(train_features,train_y)
predictions=classifier.predict(test_features)

print(accuracy_score(test_y, predictions))