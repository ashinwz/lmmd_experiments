#written by wzz，2017-3-29
#coding=utf-8
f2=open("chembl_structure.txt","w")
with open("chembl_chemical.txt","r") as f1:
	a=set()
	for line in f1.readlines():
		line=line.split("\t")
		if line[5]=="" or line[5] ==">":
			pass
		set_tuple=(line[2],line[1],line[11],line[13])
		a.add(set_tuple)
	for i in a:
                print i[2]
		f2.write("\t".join(i))
f1.close()
f2.close()		
