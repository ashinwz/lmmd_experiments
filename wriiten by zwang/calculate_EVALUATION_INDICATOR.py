#coding=utf-8
#for calculating the EVALUATION _INDICATOR
def calculate(number,name):
    with open("EVALUATION_INDICATOR.txt","r") as f:
        a=[x.strip("\n").split("\t")[number] for x in f.readlines()]
        for i in range(0,len(a)):
            if i%10==0 and i!=0:
                a[i]=str(a[i])+"|"
        auc=",".join(a)
        auc_list=auc.split("|")
        all_auc=[]
        for every in auc_list:
            every_list=every.split(",")
            every_list.pop(0)
            if every_list!=[]:
                #print every_list
                sum_list=[float(x) for x in every_list]
                a=sum(sum_list)/10
                all_auc.append(a)
                #print a
        print "-------------"
        print "%s_value:"%name,"%.4f"%max(all_auc)

if __name__=="__main__":
    calculate(1,"AUC")
    calculate(15,"P_20")
    calculate(16,"R_20")
    calculate(17,"eP_20")
    calculate(18,"eR_20")
    print "-------------"
    
            
        
