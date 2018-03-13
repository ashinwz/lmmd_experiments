from __future__ import print_function
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from  keras.layers import TimeDistributed
from sklearn.metrics import confusion_matrix,recall_score,classification_report,accuracy_score,roc_auc_score,f1_score

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)

def converse(a):
    a_list=a.tolist()
    b_list=(1-a).tolist()
    #print array_len
    a_b_list=[]
    for i in range(0,len(a_list)):
        #print i
        array_list=[]
        array_list.append(a_list[i])
        array_list.append(b_list[i])
        a_b_list.append(array_list)
    a_b_array=np.array(a_b_list)
    return a_b_array.swapaxes(1,2)


if __name__=="__main__":
    data=pd.read_csv("sort_maccs.csv")
    x=data.iloc[:,8:]
    y=data["n_np"].replace({"p":1,"n":0})
    all_inputs =x.values
    all_classes =y.values
    (x_train,x_test,y_train,y_test) = train_test_split(all_inputs, all_classes, train_size=0.80, random_state=1)
    y_train = np_utils.to_categorical(y_train,2)
    y_test = np_utils.to_categorical(y_test,2)
    x_test=np.array(x_test)
    y_test=np.array(y_test)

    x_train=converse(x_train)
    x_test=converse(x_test)

    print('Build model...')
    model = Sequential()
    model.add(LSTM(256,input_shape=(166,2),dropout=0.2,return_sequences=True))
    #model.add(Embedding(1000,128))
    #model.add(Activation('tanh'))
    model.add(LSTM(256,dropout=0.2))
    model.add(Dense(512))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(2, activation='softmax'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
    print('Train...')
    model.fit(x_train, y_train,batch_size=300,epochs=150,validation_data=(x_test, y_test),class_weight={1:0.35,0:0.65})
    pred_testing_classes =np.array(model.predict_classes(x_test))
    pred_training_classes =np.array(model.predict_classes(x_train))
    predict_prob_testing_classes = np.array(model.predict(x_test))
    predict_prob_training_classes = np.array(model.predict(x_train))
    auc_train=roc_auc_score(y_train[:,1],predict_prob_training_classes[:,1])
    cnf_matrix_train=confusion_matrix(y_train[:,1],pred_training_classes)
    print (cnf_matrix_train)
    precision_train=float(cnf_matrix_train[1,1])/(cnf_matrix_train[1,1]+cnf_matrix_train[0,1])
    sp_train=float(cnf_matrix_train[0,0])/(cnf_matrix_train[0,0]+cnf_matrix_train[0,1])
    se_train=float(cnf_matrix_train[1,1])/(cnf_matrix_train[1,1]+cnf_matrix_train[1,0])
    accuracy_train=accuracy_score(y_train[:,1],pred_training_classes)
    recall_train=recall_score(y_train[:,1],pred_training_classes)
    f1_score_train=f1_score(y_train[:,1],pred_training_classes)
    g_mean_train=np.sqrt(float(sp_train*se_train))
    print ("----------train-------" )
    print ("| %s | %s | %s |%s|   %s   |   %s   | %s |  %s   |"%("Accuracy","Precision","Recall","f1_score","SE","SP","G_mean","AUC"))
    print ("|  %0.4f  |   %0.4f  | %0.4f | %0.4f | %0.4f | %0.4f | %0.4f | %0.4f |"%(accuracy_train,precision_train,recall_train,f1_score_train,se_train,sp_train,g_mean_train,auc_train))
    print (" ")

    cnf_matrix = confusion_matrix(y_test[:,1],pred_testing_classes)
    np.set_printoptions(precision=2)
    precision_test=float(cnf_matrix[1,1])/(cnf_matrix[1,1]+cnf_matrix[0,1])
    sp_test=float(cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[0,1])
    se_test=float(cnf_matrix[1,1])/(cnf_matrix[1,1]+cnf_matrix[1,0])
    auc_test=roc_auc_score(y_test[:,1],predict_prob_testing_classes[:,1])
    accuracy_test=accuracy_score(y_test[:,1],pred_testing_classes)
    recall_test=recall_score(y_test[:,1],pred_testing_classes)
    f1_score_test=f1_score(y_test[:,1],pred_testing_classes)
    g_mean_test=np.sqrt(float(sp_test*se_test))
    print ("----------test-------")
    print ("| %s | %s | %s |%s|   %s   |   %s   | %s |  %s   |"%("Accuracy","Precision","Recall","f1_score","SE","SP","G_mean","AUC"))
    print ("|  %0.4f  |   %0.4f  | %0.4f | %0.4f | %0.4f | %0.4f | %0.4f | %0.4f |"%(accuracy_test,precision_test,recall_test,f1_score_test,se_test,sp_test,g_mean_test,auc_test))
    print (" ") 
