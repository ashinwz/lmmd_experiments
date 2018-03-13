#coding=utf-8
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import  Imputer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFECV

def create_model(neurons_1,neurons_2,optimizer="adam"):
    # create model
    model = Sequential()
    model.add(Dense(neurons_1, input_dim=166, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(neurons_2))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, init='uniform', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

if __name__=="__main__":
    data=pd.read_csv("new_BBB_exper_data_maccs.csv")
    x=data.ix[:,8:]
    y=data["n_np"].replace({"p":1,"n":0})
    all_inputs =x.values
    all_classes =y.values
    (x_train,x_test,y_train,y_test)= train_test_split(all_inputs, all_classes, train_size=0.85, random_state=1)
    y_train = np_utils.to_categorical(y_train,2)
    y_test = np_utils.to_categorical(y_test,2)
    x_test=np.array(x_test)
    y_test=np.array(y_test)
    model = KerasClassifier(build_fn=create_model,verbose=0)

    param_grid ={"batch_size":[50,100],
                        "nb_epoch":[20,50],
                         "optimizer":["SGD", "Adam"],
                         "neurons_1":[256,128],
                         "neurons_2":[128,64],
                         "class_weight":[{1:0.3,0:0.7},{1:0.2,0:0.8},{1:0.5,0:0.5}]}
    grid= GridSearchCV(estimator=model,param_grid=param_grid,cv=10,scoring="roc_auc",n_jobs=-1)
    grid_result = grid.fit(x_train, y_train)
    print  "Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
