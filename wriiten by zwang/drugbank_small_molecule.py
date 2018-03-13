#coding=utf-8
#prepare for drugbank_all_small_molecules
#written by Zwang，2017-3-27
#-------------argv1,argv2-------------#
import sys,urllib2,time
def cas2smiles(cas):
    smiles_error=''
    try:
        smiles=urllib2.urlopen('http://cactus.nci.nih.gov/chemical/structure/'+cas+'/smiles').read()
        return smiles
    except:
        smiles=''
        return smiles_error
    
f2=open(sys.argv[2],"w")
with open(sys.argv[1],"r") as f1:
    for i in f1.readlines():
        i=i.split("\t")
        smiles=cas2smiles(i[4])
	f2.write(i[0]+"\t"+i[1]+"\t"+i[2]+"\t"+smiles+"\n")
	print i[0]
	time.sleep(2)
f1.close()
f2.close()

