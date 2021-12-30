#实现基于词典的情感分析
import jieba
import pandas as pd
import os
import re
from utils import *
#载入情感词典
# 打开词典文件，返回列表
#分句
def cut_sentence(words):
    start = 0
    i = 0
    sents = []
    token=[]
    punt_list = ',.!?:;~，。！？：；～'
    for word in words:
        if word in punt_list : #检查标点符号下一个字符是否还是标点
            sents.append(words[start:i+1])
            start = i+1
            i += 1
        else:
            i += 1
            token = list(words[start:i+2]).pop() # 取下一个字符
    if start < len(words):
        sents.append(words[start:])
    return sents

#定义判断奇偶的函数
def judgeodd(num):
    if num%2==0:
        return 'even'
    else:
        return 'odd'
class Dict():
    def __init__(self,root = '/Users/puyuandong613/Downloads/SentimentAnalysis/SA_master/data/emotion_dict'):
        self.train_num = 0
        self.test_num = 0
        self.initial_dict(root)
        self.doc_length = len(self.posdict)
    def initial_dict(self,root):
        self.posdict = self.open_dict(dir='posdict',path = root)#积极情感词典
        self.negdict = self.open_dict(dir='negdict',path = root)#消极情感词典
        self.inversedict= self.open_dict(dir='inversedict',path = root)
        self.mostdict = self.open_dict(dir='mostdict',path = root)
        self.verydict= self.open_dict(dir='verydict',path = root)
        self.moredict = self.open_dict(dir='moredict',path = root)
        self.ishdict = self.open_dict(dir='ishdict',path = root)
        self.insufficientdict = self.open_dict(dir='insufficientdict',path = root)
        f=open('/Users/puyuandong613/Downloads/SentimentAnalysis/SA_master/data/emotion_dict/酒店情感词典.txt','r',encoding='utf-8')
        words = []
        value=[]
        for word in f.readlines():
            words.append(word.split(' ')[0])
            value.append(float(word.split(' ')[1].strip('\n')))
        c={'words':words,'value':value}
        fd=pd.DataFrame(c)
        pos=fd['words'][fd.value>0]
        self.posdict=self.posdict+list(pos)    ##加入酒店相关的正向情感词
        neg=fd['words'][fd.value<0]
        self.negdict=self.negdict+list(neg)    ##加入酒店相关的负向情感词
        f.close()
    def load_data(self,filepath):#添加数据
        re_split = re.compile("\s+")
        self.posdata = []
        self.negdata = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                splits = re_split.split(line.strip())
                if splits[0] == "pos":
                    self.posdata.append(splits[1:])
                elif splits[0] == "neg":
                    self.negdata.append(splits[1:])
                else:
                    raise ValueError("Corpus Error")
    def open_dict(self,dir,path='/Users/puyuandong613/Downloads/SentimentAnalysis/SA_master/data/emotion_dict'):
        # path = path + '%s.txt' %dir
        path = os.path.join(path,dir)
        path += '.txt'

        dictionary = open(path, 'r', encoding='utf-8-sig',errors='ignore')#encoding='utf-8-sig',检查是否有文件头，并去掉
        dir = []
        for word in dictionary:
            word=word.strip('\n')
            word=word.strip(' ')
            dir.append(word)
        return dir
    def get_dict(self, start=0, end=-1):
        assert self.doc_length >= self.test_num + self.train_num

        if end == -1:
            end = self.doc_length

        data = self.posdata[start:end] + self.negdata[start:end]
        data_labels = [1] * (end - start) + [0] * (end - start)
        return data, data_labels

    def get_train_dict(self, num):
        self.train_num = num
        return self.get_dict(end=num)

    def get_test_dict(self, num):
        self.test_num = num
        return self.get_dict(start=self.train_num, end=self.train_num + num)

    def get_all_dict(self):
        data = self.pos_doc_list[:] + self.neg_doc_list[:]
        data_labels = [1] * self.doc_length + [0] * self.doc_length
        return data, data_labels
#计算正、负和总的情感得分
def prediction(diction,review):
    sents=cut_sentence(review)
    #print(sents)
    pos_senti=0#段落的情感得分
    neg_senti=0
    total_senti=0
    for sent in sents:
        pos_count=0#句子的情感得分
        neg_count=0
        seg=jieba.lcut(sent,cut_all=False)
        #print(sent)
        i = 0 #记录扫描到的词的位置
        a = 0 #记录情感词的位置
        poscount = 0 #正向词的第一次分值
        poscount2 = 0 #正向词反转后的分值
        poscount3 = 0 #正向词的最后分值
        negcount = 0 #负向词的第一次分值
        negcount2 = 0 #负向词反转后的分值
        negcount3 = 0 #负向词的最后分值
        for word in seg:
            #print(word)
            poscount=0
            negcount=0
            if word in diction.posdict: #判断词语是否是情感词
                poscount += 1                
                c = 0 #情感词前否定词的个数
                for w in seg[a:i]:  #扫描情感词前的程度词
                    if w in diction.mostdict:
                        poscount *= 4.0
                    elif w in diction.verydict:
                        poscount *= 3.0
                    elif w in diction.moredict:
                        poscount *= 2.0
                    elif w in diction.ishdict:
                        poscount /= 2.0
                    elif w in diction.insufficientdict:
                        poscount /= 4.0
                    elif w in diction.inversedict:
                        c += 1
                if judgeodd(c) == 'odd': #扫描情感词前的否定词数
                    poscount *= -1.0
                    poscount2 += poscount
                    poscount = 0
                    poscount3 = poscount + poscount2 + poscount3
                    poscount2 = 0
                else:
                    poscount3 = poscount + poscount2 + poscount3
                    poscount = 0
                a = i + 1 #情感词的位置变化
            elif word in diction.negdict: #消极情感的分析，与上面一致
                negcount += 1
                d = 0
                for w in seg[a:i]:
                    if w in diction.mostdict:
                        #print(w)
                        negcount *= 4.0
                    elif w in diction.verydict:
                        #print(w)
                        negcount *= 3.0
                    elif w in diction.moredict:
                        #print(w)
                        negcount *= 2.0
                    elif w in diction.ishdict:
                        #print(w)
                        negcount /= 2.0
                    elif w in diction.insufficientdict:
                        #print(w)
                        negcount /= 4.0
                    elif w in diction.inversedict:
                        d += 1
                if judgeodd(d) == 'odd':
                    negcount *= -1.0
                    negcount2 += negcount
                    negcount = 0
                    negcount3 = negcount + negcount2 + negcount3
                    negcount2 = 0
                else:
                    negcount3 = negcount + negcount2 + negcount3
                    negcount = 0
                a = i + 1                  
            i += 1 #扫描词位置前移
        if poscount3 < 0 and negcount3 >=0:
            neg_count += negcount3 - poscount3
            pos_count = 0
        elif negcount3 < 0 and poscount3 >= 0:
            pos_count = poscount3 - negcount3
            neg_count = 0
        elif poscount3 < 0 and negcount3 < 0:
            neg_count = -poscount3
            pos_count = -negcount3
        else:
            pos_count = poscount3
            neg_count = negcount3
        #print(pos_count,neg_count)
        pos_senti=pos_senti+pos_count
        neg_senti=neg_senti+neg_count
    total_senti=pos_senti-neg_senti
    if total_senti>0:
        predictions=1
    else:
        predictions=-1
    return (predictions)
def test(test_x, test_y,diction):
    accuracy = 0  
    y_hats = []
    for sentence in test_x:
        if prediction(diction,sentence) > 0:
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
diction = Dict()
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
accuracy = test(test_x,test_y,diction)
print("accuracy:",accuracy)





