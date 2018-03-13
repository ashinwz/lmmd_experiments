from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D,MaxPooling1D
from keras.datasets import imdb
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
from sklearn.metrics import confusion_matrix,recall_score,classification_report,accuracy_score,roc_auc_score,f1_score

data=pd.read_csv("new_BBB_exper_data_maccs.csv")
x=data.iloc[:,8:]
y=data["n_np"].replace({"p":1,"n":0})
all_inputs =x.values
all_classes =y.values
(x_train,x_test,y_train,y_test) = train_test_split(all_inputs, all_classes, train_size=0.80, random_state=1)
y_train = np_utils.to_categorical(y_train,2)
y_test = np_utils.to_categorical(y_test,2)
x_test=np.array(x_test)
y_test=np.array(y_test)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(5000,128,input_length=166))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(128,18,padding='valid',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())
#model.add(MaxPooling1D(5))
#model.add(Conv1D(128, 5, activation='relu'))
#model.add(MaxPooling1D(5))
model.add(Dense(256))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,batch_size=200,epochs=30,validation_data=(x_test, y_test))
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
