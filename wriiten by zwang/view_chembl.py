#coding=utf-8
#filter for chembl_chemicals
#CHOOSE "uM,nM,microM,M,umol/L,nmol/L"
#FILEOUT: 
f2=open("chembl_filter_2.txt","w")
f1=open("chembl_filter_1.txt","w")
with open("chembl_all.txt","r") as f:
    a=f.readline().split("\t")
    f.seek(0,0)
    i=0
    chembl_set=set()
    try:
        while a[0]!="":
            a=f.readline().replace("\"","").split("\t")
            #print a
            i+=1
            if a[8] in ["uM","nM","microM","M","nmol/L","umol/L"]:
                if a[5]=="" or a[5]==">":
                    pass
                else:
                    set_tuple=(a[2],a[1],a[6],a[15])
                    if a[8]=="uM" and int(float(a[7]))<=10:
                        print "-----add--%s---"%set_tuple[0]
                        chembl_set.add(set_tuple)
                        f2.write("\t".join([a[0],a[1],a[2],a[6],a[5],a[7],a[8],a[15]]))
                    elif a[8]=="nM" and int(float(a[7]))<=100000:
                        print "-----add--%s---"%set_tuple[0]
                        chembl_set.add(set_tuple)
                        f2.write("\t".join([a[0],a[1],a[2],a[6],a[5],a[7],a[8],a[15]]))
                    elif a[8]=="M" and int(float(a[7]))<0.00001:
                        print "-----add--%s---"%set_tuple[0]
                        chembl_set.add(set_tuple)
                        f2.write("\t".join([a[0],a[1],a[2],a[6],a[5],a[7],a[8],a[15]]))
                    elif a[8]=="microM" and int(float(a[7]))<=10:
                        print "-----add--%s---"%set_tuple[0]
                        chembl_set.add(set_tuple)
                        f2.write("\t".join([a[0],a[1],a[2],a[6],a[5],a[7],a[8],a[15]]))
                    elif a[8]=="nmol/L" and int(float(a[7]))<=100000:
                        print "-----add--%s---"%set_tuple[0]
                        chembl_set.add(set_tuple)
                        f2.write("\t".join([a[0],a[1],a[2],a[6],a[5],a[7],a[8],a[15]]))
                    elif a[8]=="umol/L" and int(float(a[7]))<=10:
                        print "-----add--%s---"%set_tuple[0]
                        chembl_set.add(set_tuple)
                        f2.write("\t".join([a[0],a[1],a[2],a[6],a[5],a[7],a[8],a[15]]))
                    else:
                        pass
            else:
                pass
    except IndexError,e:
        print "bad"
    for i in chembl_set:
        f1.write("\t".join(i))
f2.close()
f1.close()
f.close()
