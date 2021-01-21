import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def evaluate(y,yhat):
#     y =y.astype('int32') 
#     yhat = yhat.astype('int32')
 
    matrix_test = confusion_matrix(y, yhat)
    
    TP = matrix_test[0][0]
    FN = matrix_test[0][1]
    FP = matrix_test[1][0]
    TN = matrix_test[1][1]

    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    print("confusion matrix",matrix_test)
    return sensitivity,specificity,accuracy

def predict(X,Xtrain,Ytrain,k):
        y_pred = [_predict(x,Xtrain,Ytrain,k) for x in X]
        return np.array(y_pred)

def _predict(x,Xtrain,Ytrain,k):
       
        distances = [euclidean_distance(x, x_train) for x_train in Xtrain]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [Ytrain[i] for i in k_idx]  
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]

def knnAlgo1(k,split_test_ratio,X,y):
      Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size = split_test_ratio, random_state = 0)
      
      predictions = predict(Xtest,Xtrain,Ytrain,k)
      acc1=accuracy(Ytest,predictions)
      senst,speci,acc=evaluate(Ytest,predictions)
      print("custom KNN classification accuracy", acc1)
      return acc,senst,speci

