#-----------------------------------IMPORTING NECESSARY LIBRARIES AND PACKAGES------------------------------------#
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
#---------------------------------LOADING DATASET AND DATA PREPROCESSING-------------------------------------------#
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
dataset=pd.get_dummies(dataset,columns=['workclass','education','marital.status','occupation','relationship','race','sex','native.country'],drop_first=True)
dataset_columns=['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 'workclass_1', 'workclass_2', 'workclass_3', 'workclass_4', 'education_1', 'education_2', 'education_3', 'education_4', 'education_5', 'education_6', 'education_7', 'education_8', 'education_9', 'education_10', 'education_11', 'education_12', 'education_13', 'education_14', 'education_15', 'marital.status_1', 'marital.status_2', 'occupation_1', 'occupation_2', 'occupation_3', 'occupation_4', 'occupation_5', 'occupation_6', 'occupation_7', 'occupation_8', 'occupation_9', 'occupation_10', 'occupation_11', 'occupation_12', 'occupation_13', 'relationship_1', 'relationship_2', 'relationship_3', 'relationship_4', 'relationship_5', 'race_1', 'race_2', 'race_3', 'race_4', 'sex_1', 'native.country_1', 'native.country_2', 'native.country_3', 'native.country_4','income']
dataset=dataset[dataset_columns]
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
lable_encoder=LabelEncoder()
Y=lable_encoder.fit_transform(Y)
Y=Y.reshape(-1,1)
enc=OneHotEncoder()
enc.fit(Y)
onehotlabels=enc.transform(Y).toarray()
Y=onehotlabels
#-----------------------------------------SPLITTING THE DATA INTO TRAINING AND TESTING---------------------------------------------#
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#----------------------------------------DEFINING VARIOUS ACTIVATION FUNCTIONS AND THEIR DERIVATIVES-------------------------------#
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

#-------------------------------------CONSTRUCTING OUR NEURAL NETWORK CLASS/STRUCTURE-----------------------------------------#
class Neural_Network:
	
	def __init__(self,x,y):
		self.input=x
		self.weights1=0.1*np.random.randn(self.input.shape[1],60)
		self.weights2=0.1*np.random.randn(60,20)
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
		self.layer1=ReLU(np.dot(self.input,self.weights1))
		self.layer2=ReLU(np.dot(self.layer1,self.weights2))
		self.output=sigmoid(np.dot(self.layer2,self.weights3))

	def BackPropogation(self):

		m=len(self.input)
		d_weights3=-(1/len(self.input))*np.dot(self.layer2.T,(self.y-self.output)*sigmoid_derivative(self.output))
		d_weights2=-(1/len(self.input))*np.dot(self.layer1.T, (np.dot((self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * ReLU_derivative(self.layer2)))
		d_weights1=-(1/len(self.input))*np.dot(self.input.T, (np.dot(np.dot((self.y-self.output) * sigmoid_derivative(self.output),self.weights3.T)*ReLU_derivative(self.layer2),self.weights2.T)*ReLU_derivative(self.layer1)))

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

	def predict(self,X):
		self.layert_1=ReLU(np.dot(X,self.weights1))
		self.layert_2=ReLU(np.dot(self.layert_1,self.weights2))
		return sigmoid(np.dot(self.layert_2,self.weights3))


#------------------------------------------{TRAINING OUR NETWORK OVER THE TRAINING DATA AND---------------------------------# 
#------------------------------------------EVALUATING VARIOUS PARAMETERS AFTER EACH EPOCH}----------------------------------#
m=len(X_train)
n=len(X_test)
epochs=500
lr=0.05

nn1=Neural_Network(X_train,Y_train)

for i in range(epochs): 
	nn1.FeedForward()
	y_predict_train=enc.inverse_transform(nn1.output.round())
	y_predict_test=enc.inverse_transform(nn1.predict(X_test).round())
	y_train=enc.inverse_transform(Y_train)
	y_test=enc.inverse_transform(Y_test)
	train_accuracy=(m-np.count_nonzero(y_train-y_predict_train))/m
	test_accuracy=(n-np.count_nonzero(y_test-y_predict_test))/n
	nn1.BackPropogation()
	cost=(1/m)*np.sum(np.square(nn1.y-nn1.output))
	print("Epoch {}/{} ==============================================================:- ".format(i+1,epochs))
	print("MSE_Cost: {} , Train_Accuracy: {} , Test_Accuracy: {} ".format(cost,train_accuracy,test_accuracy))

output=nn1.predict(X_test)
Y_predict=enc.inverse_transform(output.round())
Y_test=enc.inverse_transform(Y_test)
accuracy=(len(Y_predict)-np.count_nonzero(Y_test-Y_predict))/len(Y_predict)
print("The accuracy of the model is {}".format(accuracy))
