#coding=utf-8
#mapping for everything you want
#all file using ".tsv"
out=raw_input("please input out_filename: ")
pi=raw_input("please input your mapping filename: ")
ji=raw_input("please input your mapping number key: ")
content=raw_input("please input your mapping content number: ")
yes=raw_input("Anythingelse?(yes/no)")
if yes=="yes":
    else_content=raw_input("please input else number: ")
else:
    else_content=""
pipei=raw_input("please input your map_filename: ")
row=raw_input("please input your map row: ")
number=raw_input("please input your pipei mapping number: ")
parameter=[out,pi,ji,content,pipei,row,number]
for i in parameter:
    if i=="":
        exit()
f3=open(out,"w")
with open(pi,"r") as f1:
    aim={}
    for i in f1.readlines():
        i=i.split("\t")
        aim_index=int(ji)#匹配基
        out_index=int(content)#要匹配的内容
	if else_content!="":
	    else_index=int(else_content)#其他内容
	    if aim_index==len(i)-1:
                aim_name=str(i[aim_index]).replace("\n","")
		mapping=str(i[out_index])
		else_thing=str(i[else_index])
	    elif out_index==len(i)-1:
		mapping=str(i[out_index]).replace("\n","")
		aim_name=str(i[aim_index])
		else_thing=str(i[else_index])
            elif else_index==len(i)-1:
		mapping=str(i[out_index])
                aim_name=str(i[aim_index])
		else_thing=str(i[else_index]).replace("\n","")
	    else:
		mapping=str(i[out_index])
		aim_name=str(i[aim_index])
		else_thing=str(i[else_index])
            aim[aim_name]=[mapping,else_thing]
            #print aim
        else:
	    if aim_index==len(i)-1:
                aim_name=str(i[aim_index]).replace("\n","")
		mapping=str(i[out_index])
	    elif out_index==len(i)-1:
		mapping=str(i[out_index]).replace("\n","")
		aim_name=str(i[aim_index])
	    else:
		mapping=str(i[out_index])
		aim_name=str(i[aim_index])
	    aim[aim_name]=mapping
with open(pipei,"r") as f2:
    a=f2.readline().split("\t")
    f2.seek(0,0)
    i=0
    while i<int(row):
        #总行数 
        a=f2.readline().split("\t")
        write=[]
        for num in range(0,len(a)-1):
            write.append(a[num])
        i+=1
        print i
        pipei_index=int(number)
        if pipei_index==int(len(a)-1):
            item=a[pipei_index].replace("\n","")
            if item in aim.keys() and isinstance(aim[item],str):
                f3.write("\t".join(write)+"\t"+item+"\t"+aim[item]+"\n")
            elif item in aim.keys() and isinstance(aim[item],list):
		f3.write("\t".join(write)+"\t"+item+"\t"+aim[item][0]+"\t"+aim[item][1]+"\n")
	    else:
                pass
        else:
            item=a[pipei_index]
            if item in aim.keys() and isinstance(aim[item],str):
                f3.write("\t".join(write)+"\t"+str(a[len(a)-1]).replace("\n","")+"\t"+aim[item]+"\n")
            elif item in aim.keys() and isinstance(aim[item],list):
		f3.write("\t".join(write)+"\t"+str(a[len(a)-1]).replace("\n","")+"\t"+aim[item][0]+"\t"+aim[item][1]+"\n")
	    else:
                pass
f1.close()
f2.close()
f3.close()

