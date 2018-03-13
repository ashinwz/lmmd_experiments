#coding=utf-8
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
np.random.seed(133)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import  Imputer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFECV
import os
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix,recall_score,classification_report,accuracy_score,roc_auc_score,f1_score

def variancethreshold(finger_feature):#variance threshold
    sel_1 = VarianceThreshold(threshold=(.98 * (1 - .98))).fit_transform(finger_feature)
    #sel_2 = sel_1.get_support(indices=True)
    #finger_one=finger_feature[sel_2]
    #print "          ",finger_one.shape
    return sel_1

def f_classif_filter(finger_feature):#f-score filter
    global label
    label=data["n_np"].replace({"p":1,"n":0})
    selector_1 = SelectPercentile(f_classif, percentile=80).fit_transform(finger_feature, label)
    #selector_2 = selector_1.get_support(indices=True)
    #finger_two=finger_feature[selector_2]
    #print "          ",finger_two.shape
    return selector_1

def rfe_filter(feature_filter,finger_feature):
    from sklearn.svm import SVC
    global label
    label=data["n_np"].replace({"p":1,"n":0})
    svc=SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5), scoring='roc_auc')
    finger_three=rfecv.fit_transform(finger_feature, label)
    #rfecv_get = rfecv.get_support(indices=True)
    #finger_three=finger_feature[rfecv_get]
    print "          ",finger_three.shape
    print("Optimal number of features : %d" % rfecv.n_features_)
    return finger_three

def pca(finger_feature):
    from sklearn.decomposition import PCA
    pca=PCA().fit_transform(finger_feature) 
    print "          ",pca.shape
    return pca

def feature_selection(feature_filter,finger_feature):
    if feature_filter==1:
        finger_f=variancethreshold(finger_feature)
    elif feature_filter==2:
        finger_f_1=variancethreshold(finger_feature)
        finger_f=f_classif_filter( finger_f_1)
    elif feature_filter==3:
        finger_f_1=variancethreshold(finger_feature)
        finger_f=rfe_filter(feature_filter,finger_f_1)
    elif feature_filter==4:
        finger_f_1=variancethreshold(finger_feature)
        finger_f=pca(finger_f_1)
    elif feature_filter==5:
        finger_f_1=variancethreshold(finger_feature)
        finger_f_2=f_classif_filter( finger_f_1)
        finger_f_3=rfe_filter(feature_filter,finger_f_2)
        finger_f=pca(finger_f_3)
    else:
        finger_f=finger_feature
    return finger_f

data=pd.read_csv("All-Pub.csv")
x=data.iloc[:,1:-1]
y=data["Name"]
#finger_feature=feature_selection(1,x)
#all_inputs =finger_feature
all_inputs=x.values
all_classes =y.values
(x_train,x_test,y_train,y_test) = train_test_split(all_inputs, all_classes, train_size=0.8, random_state=1)
y_train = np_utils.to_categorical(y_train,2)
#print y_train
y_test = np_utils.to_categorical(y_test,2)
x_test=np.array(x_test)
y_test=np.array(y_test)

model = Sequential()
model.add(Dense(512, input_shape=(881,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
              metrics=['accuracy'])

model.fit(x_train, y_train,
                    nb_epoch=120,
                    verbose=1,shuffle=False, validation_data=(x_test, y_test))

pred_testing_classes =np.array(model.predict_classes(x_test))
pred_training_classes =np.array(model.predict_classes(x_train))
predict_prob_testing_classes = np.array(model.predict(x_test))
predict_prob_training_classes = np.array(model.predict(x_train))
#print predict_prob_testing_classes
#print pred_testing_classes
#print y_train[:,1]
auc_train=roc_auc_score(y_train[:,1],predict_prob_training_classes[:,1])
cnf_matrix_train=confusion_matrix(y_train[:,1],pred_training_classes)
print (" ")
print cnf_matrix_train
precision_train=float(cnf_matrix_train[1,1])/(cnf_matrix_train[1,1]+cnf_matrix_train[0,1])
sp_train=float(cnf_matrix_train[0,0])/(cnf_matrix_train[0,0]+cnf_matrix_train[0,1])
se_train=float(cnf_matrix_train[1,1])/(cnf_matrix_train[1,1]+cnf_matrix_train[1,0])
accuracy_train=accuracy_score(y_train[:,1],pred_training_classes)
recall_train=recall_score(y_train[:,1],pred_training_classes)
f1_score_train=f1_score(y_train[:,1],pred_training_classes)
g_mean_train=np.sqrt(float(sp_train*se_train))
print "----------train-------" 
print "| %s | %s | %s |%s|   %s   |   %s   | %s |  %s   |"%("Accuracy","Precision","Recall","f1_score","SE","SP","G_mean","AUC")
print "|  %0.4f  |   %0.4f  | %0.4f | %0.4f | %0.4f | %0.4f | %0.4f | %0.4f |"%(accuracy_train,precision_train,recall_train,f1_score_train,se_train,sp_train,g_mean_train,auc_train)
print " " 

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
print cnf_matrix
print "----------test-------" 
print "| %s | %s | %s |%s|   %s   |   %s   | %s |  %s   |"%("Accuracy","Precision","Recall","f1_score","SE","SP","G_mean","AUC")
print "|  %0.4f  |   %0.4f  | %0.4f | %0.4f | %0.4f | %0.4f | %0.4f | %0.4f |"%(accuracy_test,precision_test,recall_test,f1_score_test,se_test,sp_test,g_mean_test,auc_test)
print " "
sample_df=pd.DataFrame.empty
sample_df=pd.DataFrame(columns=["train_Accuracy","train_Precision","train_Recall","train_f1_score","train_SE","train_SP","train_G_mean","train_AUC",
                                 "test_Accuracy","test_Precision","test_Recall","test_f1_score","test_SE","test_SP","test_G_mean","test_AUC"])
sample_df.loc[0]={"train_Accuracy":accuracy_train,"train_Precision":precision_train,"train_Recall":recall_train,"train_f1_score":f1_score_train,"train_SE":se_train,"train_SP":sp_train,"train_G_mean":g_mean_train,"train_AUC":auc_train,
                                 "test_Accuracy":accuracy_test,"test_Precision":precision_test,"test_Recall":recall_test,"test_f1_score":f1_score_test,"test_SE":se_test,"test_SP":sp_test,"test_G_mean":g_mean_test,"test_AUC":auc_test}
sample_df.to_csv("keras_BBB.csv")
