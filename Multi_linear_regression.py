# multivariable linear regression from scratch
#Dataset1: father son height prediction


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(filename):

    df=pd.read_csv(filename)
    
    Y = list(df.columns)
    last = Y[-1]

    np.random.seed(2)

    df = df.sample(frac=1).reset_index(drop=True)

    r_no=np.random.randint(df.shape[0]-20)
    test_data= df[r_no:r_no+20]
    train_data=df.drop(range(r_no,r_no+20))

    train_output=train_data[last]
    test_output=test_data[last]
    
    train_input=train_data.drop([last],axis=1)
    test_input=test_data.drop([last],axis=1)

    test_output=np.array(test_output).reshape(1,test_output.shape[0])
    train_output=np.array(train_output).reshape(1,train_output.shape[0])
    
    return train_input,train_output,test_input,test_output,last

#this standardization results mean=0 and std=1 hence range of data would be -1 to +1
def standardize_data(X):
	X=(X-X.mean())/X.std()
	return X

#following normlization results range of data from 0 to 1
def normalize_data(X):
	mini=np.min(X)
	maxi=np.max(X)
	X=(X-mini)/(maxi-mini)
	return X


def initialize_parameters(features):
	np.random.seed(3)
	w=np.random.randn(1,features)/np.sqrt(features) #0.01
	b=0
	return w,b

def hypothesis(X,w,b):
	Z=np.dot(w,X)+b
	return Z



def gradient_descent(X,Y,Z,w,b):
	#number of samples
	m=X.shape[1]

	assert(Z.shape==Y.shape)
	#this dL/dZ where L is the cost function = (Z - Y) **2
	dZ= np.multiply(2.0,(Z-Y))

	#computing dL/dW = dL/dZ * dZ/dW, Z=W*X+b
	dw=np.dot(dZ,X.T)/m

	#computing dL/db=dZ*1
	db=np.sum(dZ,axis=1,keepdims=True)/m

	return dw,db

def update_parameters(dw,db,w,b,learning_rate):
	
	assert(dw.shape==w.shape)

	w= w-learning_rate*dw
	b= b-learning_rate*db

	return w,b

def linear_regression(X,Y,learning_rate=0.000000025,num_iterations=10001):
	#number of features
	n=X.shape[0]

	#number of samples
	m=X.shape[1]

	w,b=initialize_parameters(n)
	# costs=[]
	for i in range(num_iterations):

		Z=hypothesis(X,w,b)
		dw,db=gradient_descent(X,Y,Z,w,b)
		w,b=update_parameters(dw,db,w,b,learning_rate*(num_iterations-i))

		
	return w,b

def main(dataset,learning_rate,num_iterations):

	# learning_rate,num_iterations = hyperparameters

	X_train,Y_train,X_test,Y_test,last =load_dataset(dataset)

	test_data=X_test

	print("shape of training data: ",X_train.shape,Y_train.shape)
	print("shape of testing data: ",X_test.shape,Y_test.shape)

	X_train=normalize_data(X_train)
	X_test=normalize_data(X_test)

	w,b=linear_regression(X_train.T,Y_train,learning_rate,num_iterations)
	print()
	#predicted test samples Z_test
	#actual test samples Y_test
	Z_test=np.dot(w,X_test.T)+b
	MSE=np.sum((Z_test-Y_test)**2)/Y_test.shape[1]
	print("Testing Mean squared error is: {:.4f}".format(MSE))
	#print("Accuracy is: {:.2%}".format(1-MSE)) #round(1-MSE,2)
	print("\n\n")

	predictions=print_prediction(test_data,Y_test,Z_test,dataset,last)

	return MSE,predictions


def print_prediction(test_data,Y_test,Z_test,dataset,last):

	
	if dataset == "father_son_height.csv":
		test_data[last+" in m"]=np.squeeze(Y_test)
		test_data["Prediction for "+last+" in m"]=np.round(np.squeeze(Z_test),1)
		print(test_data)

	return test_data
	
def multilinear_Regression(no_of_iters):
	# iterations=1001
	learning_rate=0.00025
	mse,predictions=main("father_son_height.csv",learning_rate,no_of_iters)
	return mse,predictions
	


	



