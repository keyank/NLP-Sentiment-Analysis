import numpy as np
import re

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.stem import WordNetLemmatizer
ps = WordNetLemmatizer()



filename = '/Users/karthikeyan/Desktop/NLP Project/Dataset/2018-E-c-En-train.txt'
f = open(filename, 'r' )
UnPrdata = [sentence.split('\t') for sentence in f.read().split('\n')]

f = open(filename, 'r' )
data = [sentence.split('\t') for sentence in f.read().split('\n')]


for i in range(0, len(data)):
	if(len(data[i])  > 1 ):
		data[i][1] = re.sub(r'[0-9]', '', data[i][1] )
		data[i][1] = re.sub(r'[^a-zA-Z\']' , ' '  , data[i][1])
		data[i][1] = re.sub(r' +', ' ', data[i][1])
		data[i][1] = re.sub(r'\'[^a-zA-Z]', '', data[i][1] )
		data[i][1] = re.sub(r'^ ', '', data[i][1] )

ID = []
ID.append("ID")
Tweet = []
Tweet.append("Tweet")
Label = np.zeros((len(data), len(data[1][2:]) ))

for i in range(1, len(data)):
	if(len(data[i])  > 2 ):
		data[i][1] = [ps.lemmatize(word) for word in data[i][1].split(" ") ]
		ID.append( data[i][0])
		Tweet.append(data[i][1])
		Label[i] = np.asarray(data[i][2:])



flat_data = [item for sublist in Tweet for item in sublist]
Prob = Counter(flat_data)
total = sum(Prob.values())

for key in Prob:
    Prob[key] /= total



from gensim.models import KeyedVectors
filename = '/Users/karthikeyan/Desktop/Google Words/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)


a = 0.1;


sentVec = []

for i in range(0, len(Tweet)):
	sentence = Tweet[i]
	Sum_Vec = np.zeros(300)
	count  = 0
	for j in range(0, len(sentence)):
		if(sentence[j] in model.vocab ):
			count  = count + 1
			Sum_Vec = Sum_Vec +(a/(a + Prob[sentence[j]] ))*model.get_vector(sentence[j])
	if(count != 0):
		sentVec.append(Sum_Vec/count)
	else:
		print('count = 0')

print('Size of sentVec ', len(sentVec) )
print('Size of Tweets ', len(Tweet) )



from sklearn import decomposition
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(sentVec)
commonVec = pca.components_
sentVec = sentVec - commonVec






################################################################




filename = '/Users/karthikeyan/Desktop/NLP Project/Dataset/2018-E-c-En-dev.txt'
f = open(filename, 'r' )
TestUnPrdata = [sentence.split('\t') for sentence in f.read().split('\n')]

f = open(filename, 'r' )
Testdata = [sentence.split('\t') for sentence in f.read().split('\n')]


for i in range(0, len(Testdata)):
	if(len(Testdata[i])  > 1 ):
		data[i][1] = re.sub(r'[0-9]', '', data[i][1] )
		data[i][1] = re.sub(r'[^a-zA-Z\']' , ' '  , data[i][1])
		data[i][1] = re.sub(r' +', ' ', data[i][1])
		data[i][1] = re.sub(r'\'[^a-zA-Z]', '', data[i][1] )
		data[i][1] = re.sub(r'^ ', '', data[i][1] )

TestID = []
TestID.append("ID")
TestTweet = []
TestTweet.append("Tweet")
TestLabel = np.zeros((len(Testdata), len(Testdata[1][2:]) ))

for i in range(1, len(Testdata)):
	if(len(Testdata[i])  > 2 ):
		Testdata[i][1] = [ps.lemmatize(word) for word in Testdata[i][1].split(" ") ]
		TestID.append( Testdata[i][0])
		TestTweet.append(Testdata[i][1])
		TestLabel[i] = np.asarray(Testdata[i][2:])



flat_data = [item for sublist in TestTweet for item in sublist]
Prob = Counter(flat_data)
total = sum(Prob.values())

for key in Prob:
    Prob[key] /= total



from gensim.models import KeyedVectors
filename = '/Users/karthikeyan/Desktop/Google Words/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)


a = 0.1;


sentVec = []

for i in range(0, len(Tweet)):
	sentence = Tweet[i]
	Sum_Vec = np.zeros(300)
	count  = 0
	for j in range(0, len(sentence)):
		if(sentence[j] in model.vocab ):
			count  = count + 1
			Sum_Vec = Sum_Vec +(a/(a + Prob[sentence[j]] ))*model.get_vector(sentence[j])
	if(count != 0):
		sentVec.append(Sum_Vec/count)
	else:
		print('count = 0')

print('Size of sentVec ', len(sentVec) )
print('Size of Tweets ', len(Tweet) )



from sklearn import decomposition
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(sentVec)
commonVec = pca.components_
sentVec = sentVec - commonVec











