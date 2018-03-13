#written by wzz,2017-3-29
#coding=utf-8
def str2int(str):
	no_str=""
	if str=='':
		str=no_str
	else:
		if str[0]==">":
			str=no_str
		elif str[0]=="<":
			str=str.replace(str[0],"")
			str=int(float(str))
		else:
			str=int(float(str))
	return str

f2=open("BindingDB_chemicals_structure.txt","w")	
with open("BindingDB_All.tsv","r") as f1:
	bindingDB=set()
	for line in f1.readlines():
		line=line.split("\t")
		try:
			if line[36]!="1" or line[41]=="" or line[7]!="Homo sapiens":
				continue
			else:
				a=[line[8],line[9],line[10],line[11]]
				for i in a:
					i=str2int(i)
					if i=="":
						continue
					elif i<=100000:
						set_tuple=("BindingDB"+line[4],line[1],line[41])
						bindingDB.add(set_tuple)
					else:
						pass
		except IndexError,e:
			print "wrong %s" % e
	for i in bindingDB:
		print i[0]
		f2.write("\t".join(i)+"\n")
f1.close()
f2.close()
					
			
			
					
					
				
				
				
#if __name__=="__main__":
	#str2int()
