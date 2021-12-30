import numpy as np
from utils import *
from matplotlib import pyplot as plt
def sigmoid(z): 
    return 1/(1+np.exp(-z))
def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(x)
    loss_his = []
    for i in range(0, num_iters):
        z = np.dot(x,theta)
        h = sigmoid(z)
        J = -(1/m) * (np.dot(np.transpose(y), np.log(h)) +  np.dot(np.transpose(1-y), np.log(1-h)))
        theta = theta - (alpha/m)*(np.dot(np.transpose(x),(h-y)))
        loss_his.append(float(J))
    J = float(J)
    return J, theta, np.array(loss_his)
def extract_features(sentence, freqs):
# process_tweet tokenizes, stems, and removes stopwords
    word_l = process_sentence(sentence)
    x = np.zeros((1, 3)) 
    
    #bias term is set to 1
    x[0,0] = 1 
    for word in word_l:
        if (word, 1) in freqs:
            x[0,1] += freqs[(word, 1)]
        if (word, 0) in freqs:
            x[0,2] += freqs[(word, 0)]
                
    assert(x.shape == (1, 3))
    return x
def predict_sentence(tweet, freqs, theta):

    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x,theta))
    return y_pred
def test_logistic_regression(test_x, test_y, freqs, theta):
    y_hat = []
    
    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_sentence(tweet, freqs, theta)
        
        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)
    
    y_hat = np.asarray(y_hat)
    test_y = np.squeeze(test_y)

    count =0
    for i in range(len(test_y)):
        if (test_y[i] == y_hat[i]):
            count = count+ 1
        else:
            count
    accuracy = count/(len(test_y))
    return accuracy


print("Constructing dataset")
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
print("Building frequencies")
freqs = build_freqs(train_x, train_y)
print("extracting features")
X = np.zeros((len(train_x), 3))
from tqdm import  tqdm
for i in tqdm(range(len(train_x))):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta ,loss_his= gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 5000)
tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

plt.figure()
plt.plot(range(5000),loss_his)
plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

tfidf_vectorizer=TfidfVectorizer(min_df=1,norm='l2',smooth_idf=True,use_idf=True,ngram_range=(1,1))
train_features= tfidf_vectorizer.fit_transform(train_x)
test_features= tfidf_vectorizer.transform(test_x)

classifier= LogisticRegression(C=1,penalty='l2')
classifier.fit(train_features,train_y)
predictions=classifier.predict(test_features)

print(accuracy_score(test_y, predictions))