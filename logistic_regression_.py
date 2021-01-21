import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression
#from regression import LogisticRegression

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

def logistic_model(X,y,learning_rate, no_of_iterations, test_split_ratio= 0.2):

    # bc = datasets.load_breast_cancer()
    # X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio, random_state=1234)

    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    senst,speci,acc=evaluate(y_test, predictions)
    acc=accuracy(y_test, predictions)
    print("LR classification accuracy:",acc )

    return senst,speci,acc