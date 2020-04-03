import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data=pd.read_csv('IRIS.csv')

X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values
StandardScaler=StandardScaler()
X=StandardScaler.fit_transform(X)

label_encoder=LabelEncoder()
Y=label_encoder.fit_transform(Y)
Y=Y.reshape(-1,1)
enc=preprocessing.OneHotEncoder()
enc.fit(Y)
onehotlabels=enc.transform(Y).toarray()
Y=onehotlabels
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
def ReLU(x):
	return (abs(x.astype(float))+x.astype(float))/2

def ReLU_derivative(x):
	y=x
	np.piecewise(y,[ReLU(y)==0,ReLU(y)==y],[0,1])
	return y

def tanh(x):
	return np.tanh(x.astype(float))

def tanh_derivative(x):
	return 1-np.square(tanh(x))

def sigmoid(x):
	return 1/(1+np.exp(-x.astype(float)))

def sigmoid_derivative(x):
	return x*(1-x)

class Neural_Network:
	def __init__(self,x,y):
		self.input=x
		self.weights1=np.random.randn(self.input.shape[1],10)
		self.weights2=np.random.randn(10,3)
		self.y=y
		self.output=np.zeros(y.shape)

	def FeedForward(self):
		self.layer1=ReLU(np.dot(self.input,self.weights1))
		self.output=sigmoid(np.dot(self.layer1,self.weights2))

	def BackPropogation(self):
		lr=2
		m=len(self.input)
		d_weights2=-(1/m)*np.dot(self.layer1.T,(self.y-self.output)*sigmoid_derivative(self.output))
		d_weights1 =-(1/m)*np.dot(self.input.T, (np.dot((self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * ReLU_derivative(self.layer1)))
		self.weights2=self.weights2 - lr*d_weights2
		self.weights1=self.weights1 - lr*d_weights1
	def predict(self,X):
		self.input=X
		self.layer1=ReLU(np.dot(self.input,self.weights1))
		self.output=sigmoid(np.dot(self.layer1,self.weights2))

epochs=10000
m=len(X)
nn1=Neural_Network(X_train,Y_train)
for i in range(epochs):
	nn1.FeedForward()
	nn1.BackPropogation()
	cost=(1/m)*np.sum(np.square(nn1.y-nn1.output))
	print("cost after iteration {} : {}".format(i,cost))
nn1.predict(X_test)
Y_predict=enc.inverse_transform(nn1.output.round())
Y_test=enc.inverse_transform(Y_test)
temp=Y_test-Y_predict
accuracy=(len(Y_predict)-np.count_nonzero(temp))/len(Y_predict)
print("The accuracy of the model in {}".format(accuracy))

