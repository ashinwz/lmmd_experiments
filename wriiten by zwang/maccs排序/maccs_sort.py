#coding=utf-8

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

if __name__=="__main__":
    data=pd.read_csv("new_BBB_exper_data_maccs.csv")
    x=data.iloc[:,8:].T
    y=data["n_np"].replace({"p":1,"n":0})
    model = AgglomerativeClustering(n_clusters=10,linkage="average", affinity="euclidean")
    y_pref=model.fit_predict(x)
    print y_pref
    print x.index[:]
    maccs_list=[]
    for i in range(0,166):
        tuple_maccs=(x.index[i],y_pref[i])
        maccs_list.append(tuple_maccs)
    sort_maccs_list=sorted(maccs_list,key=lambda x:x[1])
    print len(sort_maccs_list)
    print sort_maccs_list
    new_data=pd.DataFrame()
    for i in range(0,166):
        new_data[sort_maccs_list[i][0]]=data[sort_maccs_list[i][0]]
    new_data.to_csv("sort_maccs.csv")

