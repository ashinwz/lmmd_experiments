#coding=utf-8
import numpy as np

def converse(a):
	a_list=a.tolist()
	b_list=(1-a).tolist()
	#print array_len
	a_b_list=[]
	for i in range(0,len(a_list)):
		print i
		array_list=[]
		array_list.append(a_list[i])
		array_list.append(b_list[i])
		a_b_list.append(array_list)
	a_b_array=np.array(a_b_list)
	return a_b_array.swapaxes(1,2)

if __name__=="__main__":
	a=np.array([[1,0,0,0,1],
				[0,1,1,1,1],
				[1,0,0,0,0]])
	array=converse(a)
	print array

