import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt




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

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


n_samples, n_features = X_train.shape
_classes = np.unique(y_train)
n_classes = len(_classes)

        # calculate mean, var, and prior for each class
_mean = np.zeros((n_classes, n_features), dtype=np.float64)
_var = np.zeros((n_classes, n_features), dtype=np.float64)
_priors =  np.zeros(n_classes, dtype=np.float64)

for idx, c in enumerate(_classes):
        X_c = X[y==c]
        _mean[idx, :] = X_c.mean(axis=0)
        _var[idx, :] = X_c.var(axis=0)
        _priors[idx] = X_c.shape[0] / float(n_samples)

def predict(X):
        y_pred = [_predict(x) for x in X]
        return np.array(y_pred)

def _predict(x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(_classes):
            prior = np.log(_priors[idx])
            posterior = np.sum(np.log(_pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
            
        # return class with highest posterior probability
        return _classes[np.argmax(posteriors)]
            

def _pdf(class_idx, x):
        mean = _mean[class_idx]
        var = _var[class_idx]
        numerator = np.exp(- (x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

def Naive_Bayes_Model(test_size):
    predictions = predict(X_test)
    senst,spec,acc=evaluate(y_test,predictions)
    return senst,spec,acc