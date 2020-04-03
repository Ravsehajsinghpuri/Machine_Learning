import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

dataset=pd.read_csv('adult.csv')

dataset=dataset[dataset["workclass"]!='?']
dataset=dataset[dataset["occupation"]!='?']
dataset=dataset[dataset["native.country"]!='?']
dataset.loc[dataset["workclass"]=="Without-pay","workclass"]="unemployed"

dataset.loc[dataset["workclass"]=="Self-emp-inc","workclass"]="self-employed"
dataset.loc[dataset["workclass"]=="Self-emp-not-inc","workclass"]="self-employed"
dataset.loc[dataset["workclass"]=="Local-gov","workclass"]="SL-gov"
dataset.loc[dataset["workclass"]=="State-gov","workclass"]="SL-gov"
dataset.loc[dataset["marital.status"]=="Married-civ-spouse","marital.status"]="Married"
dataset.loc[dataset["marital.status"]=="Married-AF-spouse","marital.status"]="Married"
dataset.loc[dataset["marital.status"]=="Married-spouse-absent","marital.status"]="Married"
dataset.loc[dataset["marital.status"]=="Divorced","marital.status"]="Not-Married"
dataset.loc[dataset["marital.status"]=="Separated","marital.status"]="Not-Married"
dataset.loc[dataset["marital.status"]=="Widowed","marital.status"]="Not-Married"
North_America=["United-States","Mexico","Canada","Dominican-Republic","El-Salvador","Guatemala","Haiti","Honduras","Jamaica","Puerto-Rico","Trinadad&Tobago","Outlying-US(Guam-USVI-etc)","Cuba","Nicaragua"]
Asia=["Cambodia","China","Hong","India","Iran","Japan","Laos","Philippines","Taiwan","Thailand","Vietnam"]
South_America=["Columbia","Ecuador","Peru"]
Europe=["England", "France", "Germany", "Greece", "Holand-Netherlands","Hungary", "Ireland", "Italy", "Poland", "Portugal", "Scotland","Yugoslavia"]
Other=["south"]
dataset.loc[dataset["native.country"].isin(North_America),"native.country"]="North America"
dataset.loc[dataset["native.country"].isin(Asia),"native.country"]="Asia"
dataset.loc[dataset["native.country"].isin(South_America),"native.country"]="South America"
dataset.loc[dataset["native.country"].isin(Europe),"native.country"]="Europe"
dataset.loc[dataset["native.country"].isin(Other),"native.country"]="Other"

scaler=MinMaxScaler()
numerical=['age','education.num','capital.loss','capital.gain','hours.per.week']
dataset[numerical]=scaler.fit_transform(dataset[numerical])
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

dataset=MultiColumnLabelEncoder(columns=['workclass','education','marital.status','occupation','relationship','race','sex','native.country']).fit_transform(dataset)

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
columns=[1,3,5,6,7,8,9,13]
for col in columns:
	columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(),[col])], remainder='passthrough')
	X=np.array(columnTransformer.fit_transform(X),dtype='int')
print(X[1])
lable_encoder=LabelEncoder()
Y=lable_encoder.fit_transform(Y)
Y=Y.reshape(-1,1)
enc=OneHotEncoder()
enc.fit(Y)
onehotlabels=enc.transform(Y).toarray()
Y=onehotlabels
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
'''
epochs=1000
lr=0.1

def ReLU(x):
	return (abs(x.astype(float))+x.astype(float))/2

def ReLU_derivative(x):
	y=x
	np.piecewise(y,[ReLU(y)==0,ReLU(y)==y],[0,1])
	return y

def tanh(x):
	return np.tanh(x)

def tanh_derivative(x):
	return 1-np.square(tanh(x))

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
	return x*(1-x)
def softmax(x):
	return np.exp(x/sum(np.exp(x)))

class Neural_Network:
	
	def __init__(self,x,y):
		self.input=x
		self.weights1=0.1*np.random.randn(self.input.shape[1],40)
		self.weights2=0.1*np.random.randn(40,20)
		self.weights3=0.1*np.random.randn(20,2)
		self.y=y
		self.output=np.zeros(y.shape)
		self.first_moment1=0
		self.first_moment2=0
		self.first_moment3=0
		self.second_moment1=0
		self.second_moment2=0
		self.second_moment3=0
		self.beta1_1=0.9
		self.beta1_2=0.9
		self.beta1_3=0.9
		self.beta2_1=0.999
		self.beta2_2=0.999
		self.beta2_3=0.999

	def FeedForward(self):
		self.layer1=sigmoid(np.dot(self.input,self.weights1))
		self.layer2=sigmoid(np.dot(self.layer1,self.weights2))
		self.output=sigmoid(np.dot(self.layer2,self.weights3))

	def BackPropogation(self):

		m=len(self.input)
		d_weights3=-(1/len(self.input))*np.dot(self.layer2.T,(self.y-self.output)*sigmoid_derivative(self.output))
		d_weights2=-(1/len(self.input))*np.dot(self.layer1.T, (np.dot((self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)))
		d_weights1=-(1/len(self.input))*np.dot(self.input.T, (np.dot(np.dot((self.y-self.output) * sigmoid_derivative(self.output),self.weights3.T)*sigmoid_derivative(self.layer2),self.weights2.T)*sigmoid_derivative(self.layer1)))

		self.first_moment3=self.beta1_3*self.first_moment3 + (1-self.beta1_3)*d_weights3
		self.second_moment3=self.beta2_3*self.second_moment3 + (1-self.beta2_3)*d_weights3*d_weights3
		self.first_unbias3=self.first_moment3/(1-self.beta1_3**(i+1))
		self.second_unbias3=self.second_moment3/(1-self.beta2_3**(i+1))
		self.weights3=self.weights3 - lr*(self.first_unbias3/(np.sqrt(self.second_unbias3)+1e-3))


		self.first_moment2=self.beta1_2*self.first_moment2 + (1-self.beta1_2)*d_weights2
		self.second_moment2=self.beta2_2*self.second_moment2 + (1-self.beta2_2)*d_weights2*d_weights2
		self.first_unbias2=self.first_moment2/(1-self.beta1_2**(i+1))
		self.second_unbias2=self.second_moment2/(1-self.beta2_2**(i+1))
		self.weights2=self.weights2 - lr*(self.first_unbias2/(np.sqrt(self.second_unbias2)+1e-3))


		self.first_moment1=self.beta1_1*self.first_moment1 + (1-self.beta1_1)*d_weights1
		self.second_moment1=self.beta2_1*self.second_moment1 + (1-self.beta2_1)*d_weights1*d_weights1
		self.first_unbias1=self.first_moment1/(1-self.beta1_1**(i+1))
		self.second_unbias1=self.second_moment1/(1-self.beta2_1**(i+1))
		self.weights1=self.weights1 - lr*(self.first_unbias1/(np.sqrt(self.second_unbias1)+1e-3))
	
	def Train(self):
		batch_size=200
		num_batch=int(np.round(m/batch_size))
		for i in range(num_batch):
			if(i==num_batch-1):
				self.input=self.input[batch_size*(i):,:]
				self.y=self.y[batch_size*(i):,:]
				self.FeedForward()
				self.BackPropogation()
				break
			self.input=self.input[batch_size*(i):batch_size*(i+1),:]
			print(self.input)
			self.y=self.y[batch_size*(i):batch_size*(i+1),:]
			print(self.y)
			self.FeedForward()
			self.BackPropogation()

	def predict(self,X):
		self.input=X
		self.layer1=sigmoid(np.dot(self.input,self.weights1))
		self.layer2=sigmoid(np.dot(self.layer1,self.weights2))
		self.output=sigmoid(np.dot(self.layer2,self.weights3))


m=len(X_train)

nn1=Neural_Network(X_train,Y_train)
for i in range(epochs): 
	nn1.Train()
	cost=(1/m)*np.sum(np.square(nn1.y-nn1.output))
	print("cost after iteration {} : {}".format(i,cost))
nn1.predict(X_test)
Y_predict=enc.inverse_transform(nn1.output.round())
Y_test=enc.inverse_transform(Y_test)
temp=Y_test-Y_predict
accuracy=(len(Y_predict)-np.count_nonzero(temp))/len(Y_predict)
print("The accuracy of the model in {}".format(accuracy))

'''



