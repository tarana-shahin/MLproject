import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
import knn as ks
from sklearn import datasets
import linear_regression_ as linear_regrr
import perceptron_ as single_per
import logistic_regression_ as lgr
import Neural_Networks as dnn
import Multi_linear_regression as multil
import naivebayes as nb
import KMeans as kmeans
import decision_tree_ as dt
import load_css as lcss
import base64

main_bg = "sample.jpg"
main_bg_ext = "jpg"

side_bg = "sample.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)



lcss.local_css("style.css")
 
t = "<div><span class='highlight red'> <span class='bold'>Machine Learning GUI Project</span></span></div>"

st.markdown(t, unsafe_allow_html=True)




learning_option = st.sidebar.selectbox(
    'Select Learning Algorithm',
    ('Supervised Learning', 'Unsupervised Learning')
)



def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine(return_X_y=True)
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y



def get_learning_option( learning_option ) :
    option = None
    if( learning_option == 'Supervised Learning') :
        option = st.sidebar.selectbox(
                'Select Model',
                    ('Classification', 'Regression'))
        
    else :
        option = st.sidebar.selectbox('Model',
                    ('K-Means', ''))
            
    return option
        




def get_option(name):
    
    if name == 'Classification':
        classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'Single Perceptron','Logistic Regression','Multi Layer Perceptron','Decision Tree','Naive Bayes')
)

    elif name == 'Regression':
        classifier_name = st.sidebar.selectbox('Select Model',('Simple Linear Regression', 'Multivariate Linear Regression'))
    else:
        #K means cluster
        classifier_name = 'kmeans'
    return classifier_name




def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'Single Perceptron' :
        lr = (int)(st.sidebar.number_input('Enter Learning Rate( 0.0 - 1.0 ) :  ') )
        params['lr'] = lr
        
       

        K = st.sidebar.slider('Enter the no of Iterations', 1, 5000)
        params['no_of_itr'] = K

        
        test_split_ratio = (st.sidebar.number_input('Enter Test Split Ratio : ') )
        params['test_split_ratio'] = test_split_ratio

    elif clf_name == 'Simple Linear Regression' :
        lr = (int)(st.sidebar.number_input('Enter Learning Rate( 0.0 - 1.0 ) :  ') )
        params['lr'] = lr
        
       

        K = st.sidebar.slider('Enter the no of Iterations', 0, 5000)
        params['no_of_itr'] = K

        
        test_split_ratio = (st.sidebar.number_input('Enter Test Split Ratio : ') )
        params['test_split_ratio'] = test_split_ratio


    
    elif clf_name == 'Logistic Regression':
        lr = (int)(st.sidebar.number_input('Enter Learning Rate( 0.0 - 1.0 ) :  ') )
        params['lr'] = lr
        
     

        K = st.sidebar.slider('Enter the no of Iterations', 0, 5000)
        params['no_of_itr'] = K

        test_split_ratio = (st.sidebar.number_input('Enter Test Split Ratio : ') )
        params['test_split_ratio'] = test_split_ratio
    
    elif clf_name == 'Multivariate Linear Regression':
        
        K = st.sidebar.slider('Enter the no of Iterations', 0, 5000)
        params['no_of_itr'] = K


    elif clf_name == 'KNN' :
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K

        test_split_ratio = (st.sidebar.number_input('Enter Test Split Ratio : ') )
        params['test_split_ratio'] = test_split_ratio
    
    elif clf_name == 'kmeans':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
        no_of_itr = st.sidebar.slider('Enter the no of Iterations', 0, 5000)
        params['no_of_itr'] = no_of_itr

    elif clf_name == 'Multi Layer Perceptron':
        test_split_ratio = (st.sidebar.number_input('Enter Test Split Ratio : ') )
        params['test_split_ratio'] = test_split_ratio
        hidden = st.text_input('Enter the hidden layers :')
        params['hidden_layers']=hidden
        learning_rates = st.text_input('Enter the learning rate :')
        params['learning_rate']=learning_rates
        K = st.sidebar.slider('Enter the no of Iterations', 0, 5000)
        params['no_of_itr'] = K

    elif clf_name == 'Naive Bayes' :
       
        test_split_ratio = (st.sidebar.number_input('Enter Test Split Ratio : ') )
        params['test_split_ratio'] = test_split_ratio
    
    elif clf_name == 'Decision Tree' :
       
        test_split_ratio = (st.sidebar.number_input('Enter Test Split Ratio : ') )
        params['test_split_ratio'] = test_split_ratio



    
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

def get_model( classifer_name)  :
    if classifer_name == 'KNN' :
        params = add_parameter_ui(classifier_name)
       
        accu,sensti,speci= ks.knnAlgo1(params['K'], params['test_split_ratio'],X1,Y1)
        st.write('The Accuracy of the Model is:',accu)
        st.write(' The Senstivity of the Model is : ', sensti)
        st.write(' The Specificity of the Model is : ', speci)
       
    
    elif classifer_name == 'Single Perceptron' :
        params = add_parameter_ui(classifier_name)
        y_pred, accu, pltt =  single_per.perceptron_model(params['lr'], params['no_of_itr'], params['test_split_ratio'])
       
        st.write(' The Predicted Values are  : ', y_pred, '\n')
        st.write(' The Accuracy of the Model is : ', accu)
        st.write(' The Graph is : \n', st.pyplot( pltt ) )
    

    elif classifer_name == 'Simple Linear Regression' :
        params = add_parameter_ui(classifier_name)
        predictions, rmse, pltt  = linear_regrr.linearRegression_Model(params['lr'], params['no_of_itr'], params['test_split_ratio'])
       
        st.write(' The predictions are : ' , predictions )
        st.write(' The Root Mean Square Error of the Model is : ',rmse, '\n')
        st.write(' The Graph is : \n', st.pyplot( pltt ) )

    
    elif classifer_name == 'Multivariate Linear Regression' :
        params = add_parameter_ui(classifier_name)
        mse,predictions =  multil.multilinear_Regression(params['no_of_itr'])
       
        st.write(' The Predicted Values are  : ', predictions, '\n')
        st.write(' The Mean Square Error of the Model is : ', mse)
    
        
    elif classifer_name == 'Logistic Regression' :
        params = add_parameter_ui(classifier_name)
        senst,speci,accu = lgr.logistic_model(X1,Y1,params['lr'], params['no_of_itr'], params['test_split_ratio'])
        st.write(' The Accuracy of the Model is : ', accu)
        st.write(' The Senstivity of the Model is : ', senst)
        st.write(' The Specificity of the Model is : ', speci)
    
    elif classifer_name == 'kmeans' :
        st.set_option('deprecation.showPyplotGlobalUse', False)
        params = add_parameter_ui(classifier_name)
        X, y_pred, clusters, centroid, pltt = kmeans.k_means(params['K'], params['no_of_itr'])
       
        st.write( ' The centroids  : ', centroid)
        st.pyplot( pltt)
    
    elif classifer_name == 'Naive Bayes' :
        
        params = add_parameter_ui(classifier_name)
        senst,speci,acc= nb.Naive_Bayes_Model(params['test_split_ratio'])
        
        st.write( ' The Accuracy is  : ', acc)
        st.write(' The Senstivity of the Model is : ', senst)
        st.write(' The Specificity of the Model is : ', speci)
    
    elif classifer_name == 'Decision Tree' :
        
        params = add_parameter_ui(classifier_name)
        senst,spec,acc= dt.Decision_Tree_Model(X1,Y1,params['test_split_ratio'])
        
        st.write( ' The Accuracy is  : ', acc)
        st.write(' The Senstivity of the Model is : ', senst)
        st.write(' The Specificity of the Model is : ', spec)
        

    elif classifer_name=='Multi Layer Perceptron':
        params=add_parameter_ui(classifer_name)
        costs,acc,senst,speci=dnn.L_layer_model(X1,Y1,params['hidden_layers'],params['learning_rate'],params['no_of_itr'],params['test_split_ratio'])
        st.write(' The cost will be  : ', costs, '\n')
        
        st.write(' The Accuracy of the Model is : ', accu)
        st.write(' The Senstivity of the Model is : ', sensti)
        st.write(' The Specificity of the Model is : ', speci)



option = get_learning_option( learning_option)
classifier_name = get_option(option)
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer','Wine')
)
X1, Y1 = get_dataset(dataset_name)
st.write('Shape of dataset:', X1.shape)
st.write('number of classes:', len(np.unique(Y1)))



st.subheader(f"**Learning Algorithm Type :** {learning_option} ")
st.subheader(f"**Model Type :** {classifier_name}")


get_model(classifier_name)


