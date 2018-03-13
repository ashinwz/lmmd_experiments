import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import  Imputer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC as svc
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import train_test_split
from imblearn.under_sampling import RandomUnderSampler as RU
from imblearn.over_sampling import SMOTE as SM
from imblearn.over_sampling import ADASYN as AD
from imblearn.combine import SMOTEENN as SE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.neighbors import KNeighborsClassifier as knn 
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import AdaBoostClassifier as adboost

def read_file(filename):#read fingerprints
    data = pd.read_csv(filename,low_memory=False,error_bad_lines=False)
    finger_feature = data.ix[:,8:]
    return data, finger_feature

def data_process_nan(finger_feature):# filter NaN 
    imp=Imputer(missing_values='NaN',strategy='median',axis=0)
    finger_filter_feature=imp.fit(finger_feature)
    return finger_filter_feature
    
def count(data): # count the value of label
    global label
    label=data["n_np"].replace({"p":1,"n":-1})
    num_label=data["n_np"].value_counts()
    return num_label

#----------feature selection------------#
def variancethreshold(finger_feature):#variance threshold
    sel_1 = VarianceThreshold(threshold=(.98 * (1 - .98))).fit(finger_feature)
    sel_2 = sel_1.get_support(indices=True)
    finger_one=finger_feature[sel_2]
    print "          ",finger_one.shape
    return finger_one

def f_classif_filter(finger_feature):#f-score filter
    selector_1 = SelectPercentile(f_classif, percentile=80).fit(finger_feature, label)
    selector_2 = selector_1.get_support(indices=True)
    finger_two=finger_feature[selector_2]
    print "          ",finger_two.shape
    return finger_two

def rfe_filter(feature_filter,finger_name,finger_feature):
    from sklearn.svm import SVC
    svc=SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5), scoring='roc_auc')
    rfecv.fit(finger_feature, label)
    rfecv_get = rfecv.get_support(indices=True)
    finger_three=finger_feature[rfecv_get]
    print "          ",finger_three.shape
    print("Optimal number of features : %d" % rfecv.n_features_)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    path_2= unicode("C:\Users\Administrator\Desktop\BBB_database_2\指纹数据\特征图片","utf-8")
    os.chdir(path_2)
    save_name=str(feature_filter)+"_"+str(rfecv.n_features_)+"_"+finger_name+"_"+"rfe.jpg"
    if save_name in os.listdir(path_2):
        save_name=str(feature_filter)+"_"+str(rfecv.n_features_)+"_"+finger_name+"_"+"pca"+"_"+"rfe.jpg"
    plt.savefig(save_name)
    path = unicode("C:\Users\Administrator\Desktop\BBB_database_2\指纹数据\多个指纹","utf-8")
    os.chdir(path)
    return finger_three

def pca(finger_feature):
    from sklearn.decomposition import PCA
    pca=PCA().fit_transform(finger_feature) 
    print "          ",pca.shape
    return pca

def feature_selection(feature_filter,finger_name,finger_feature):
    if feature_filter==1:
        finger_f=variancethreshold(finger_feature)
    elif feature_filter==2:
        finger_f_1=variancethreshold(finger_feature)
        finger_f=f_classif_filter( finger_f_1)
    elif feature_filter==3:
        finger_f_1=variancethreshold(finger_feature)
        finger_f=rfe_filter(feature_filter,finger_name,finger_f_1)
    elif feature_filter==4:
        finger_f_1=variancethreshold(finger_feature)
        finger_f=pca(finger_f_1)
    elif feature_filter==5:
        finger_f_1=variancethreshold(finger_feature)
        finger_f_2=f_classif_filter( finger_f_1)
        finger_f_3=rfe_filter(feature_filter,finger_name,finger_f_2)
        finger_f=finger_f_3
    else:
        finger_f=finger_feature
    return finger_f
        
#----------generation of sampling-------------#
def sampling(model,training_inputs,training_classes):
    random_state=5
    if model !="":
        sample_model=model(random_state=random_state)
        x_train_sample,y_train_sample=sample_model.fit_sample(training_inputs,training_classes)
    else:
        x_train_sample,y_train_sample=training_inputs,training_classes
    return x_train_sample,y_train_sample

#----------predict model---------------------#
def predict(finger_name,finger_filter,method,training_inputs,training_classes):
    sample_df=pd.DataFrame.empty
    sample_df=pd.DataFrame(columns=["Accuracy","Precision","Recall","f1_score","SE","SP","G_mean","AUC"])
    random_state=0
    if method.__name__=="SVC":
        predict_model=method(random_state=random_state,probability=True)
    elif method.__name__=="KNeighborsClassifier" :
        predict_model=method()
    else:
        predict_model=method(random_state=random_state)

    sample_model=[SM,SE,AD,RU,""]
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import make_scorer
    def sp(x,y):
        cm=confusion_matrix(x,y)
        sp=float(cm[0,0])/(cm[0,0]+cm[0,1])
        return sp

    def se(x,y):
        cm=confusion_matrix(x,y)
        se=float(cm[1,1])/(cm[1,1]+cm[1,0])
        return se

    def G_mean(x,y):
        sp_g=sp(x,y)
        se_g=se(x,y)
        g=np.sqrt(sp_g*se_g)
        return g

    sp_score = make_scorer(sp, greater_is_better=True)
    se_score = make_scorer(se, greater_is_better=True)
    g_score=make_scorer(G_mean,greater_is_better=True)
    for m in sample_model:
        if m=="":
            print "----------! No sampling---------"
            text="none"+"_"+method.__name__
        else:
            print "----------%s----------"%m.__name__
            text=m.__name__+"_"+method.__name__
        x_train_sample,y_train_sample = sampling(m,training_inputs,training_classes)
        scores_1 = cross_val_score(predict_model,x_train_sample,y_train_sample, cv=5,scoring="accuracy").mean()
        scores_2 = cross_val_score(predict_model,x_train_sample,y_train_sample,cv=5,scoring="roc_auc").mean()
        scores_3 = cross_val_score(predict_model,x_train_sample,y_train_sample, cv=5,scoring="f1").mean()
        scores_4 = cross_val_score(predict_model,x_train_sample,y_train_sample, cv=5,scoring="precision").mean()
        scores_5 = cross_val_score(predict_model,x_train_sample,y_train_sample, cv=5,scoring=sp_score).mean()
        scores_6 = cross_val_score(predict_model,x_train_sample,y_train_sample, cv=5,scoring=se_score).mean()
        scores_7 = cross_val_score(predict_model,x_train_sample,y_train_sample, cv=5,scoring=g_score).mean()
        scores_8 = cross_val_score(predict_model,x_train_sample,y_train_sample, cv=5,scoring="recall").mean()
        sample_df.loc[text]={"Accuracy":scores_1,"Precision":scores_4,"Recall":scores_8,"f1_score":scores_3,"SE":scores_6,"SP":scores_5,"G_mean":scores_7,"AUC":scores_2}
    path_2= unicode("C:\Users\Administrator\Desktop\BBB_database_2\指纹数据\指纹结果","utf-8")
    os.chdir(path_2)
    sample_df.to_csv("%s_%s_%d.csv"%(finger_name,method.__name__,finger_filter))
    print "~~~$$$$    %s has been done   $$$$~~~"%method.__name__
    print ""
    print ""

#----------choose the best fingerprint----------#


if __name__=="__main__":
    path = unicode("C:\Users\Administrator\Desktop\BBB_database_2\指纹数据\多个指纹","utf-8")
    os.chdir(path)
    predict_method=[rf,lr,svc,knn,mlp,adboost]
    finger_filter=[1,2,3,4,5,0]
    li = os.listdir(path)[0:-1]
    for i in li:
        try:
            if ".csv" in i:
                #read fingerprints
                path = unicode("C:\Users\Administrator\Desktop\BBB_database_2\指纹数据\多个指纹","utf-8")
                os.chdir(path)
                read_data, read_finger_feature=read_file(i)
                finger_name=i[19: i.rfind(".")]
                print " "
                print ""
                print i
                print count(read_data)
                try:
                    for f in finger_filter:
                        print "***--------%d    %s---------***"%(f,finger_name)
                        finger_f=feature_selection(f,finger_name,read_finger_feature)
                        #generate test and train
                        all_inputs =finger_f
                        all_classes =label.values
                        training_inputs,testing_inputs,training_classes,testing_classes = train_test_split(all_inputs, all_classes, train_size=0.80, random_state=1)
                        for m in predict_method:
                            print "#------------------------------%s--------------------------------#"%m.__name__
                            predict(finger_name,f,m,training_inputs,training_classes)
                            path = unicode("C:\Users\Administrator\Desktop\BBB_database_2\指纹数据\多个指纹","utf-8")
                            os.chdir(path)
                except Exception,e:
                    print e
        except Exception,e:
            print e
    
    
    
