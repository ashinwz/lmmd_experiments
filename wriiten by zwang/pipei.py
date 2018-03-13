#coding=utf-8
#mapping for everything you want
#all file using ".tsv"
out=raw_input("please input out_filename: ")
pi=raw_input("please input your mapping filename: ")
ji=raw_input("please input your mapping number key: ")
content=raw_input("please input your mapping content number: ")
pipei=raw_input("please input your map_filename: ")
row=raw_input("please input your map row: ")
number=raw_input("please input your %s mapping number: ")%pipei
parameter=[out,pi,ji,content,pipei,row,number]
for i in parameter:
    if i=="":
        exit()
f3=open(a,"w")
with open(pi,"r") as f1:
    aim={}
    for i in f1.readlines():
        i=i.split("\t")
        aim_index=int(ji)#匹配基
        out_index=int(content)#要匹配的内容
        if out_index==len(i)+1:
            mapping=str(i[out_index]).replace("\n","")
        else:
            mapping=str(i[out_index])
        aim[i[aim_index]]=mapping
with open(pipei,"r") as f2:
    a=f2.readline().split("\t")
    write=[]
    for x in range(0,len(a)-1):
        write,append(a[x])
    f2.seek(0,0)
    i=0
    while i<int(row):
        #总行数 
        a=f2.readline().split("\t")
        i+=1
        print i
        pipei_index=int(number)
        if a[pipei_index] in aim.keys():
            f3.write("\t".join(write)+"\t"+a[len(a)-1]+"\t"+aim[number]+"\n")
        else:
            pass
f1.close()
f2.close()
f3.close()

