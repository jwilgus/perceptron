### simple 2-D linear classifier 
import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt  
import time 

#--- variables
arb_iter = 100
xline = np.arange(0,101);
w_now = np.array([0.1, 0.1, 0.1]); # if [0 0 0] then decision surface wont totally seperate points

#--- generate data
x1 = rand.randint(1,50,50)
y1 = rand.randint(1,50,50)
x2 = rand.randint(50,100,50)
y2 = rand.randint(50,100,50)

#--- label data and shuffle 
class0 = np.zeros((1,50))
class1 = np.ones((1,50))
class_labels = np.concatenate((class0,class1),axis=100)

inputs = [np.concatenate((x1,x2),axis=0),
          np.concatenate((y1,y2),axis=0),
          np.concatenate(np.ones((1,100)),axis=100),
          class_labels]
print(inputs)

idx = rand.permutation(100)
D = [inputs[0][idx],
     inputs[1][idx],
     inputs[2][idx],
     inputs[3][idx]]
D = np.array(D)
print(D)



#--- linearly classify data point clusters
 
all_error = []

for s in range(arb_iter):
	each_error = [] 
	for idx in range(len(D[0])):
		print(w_now*np.array([D[0][idx], D[1][idx], D[2][idx]]))
		actual_output = w_now@np.array([D[0][idx], D[1][idx], D[2][idx]])
		print(actual_output)
		
		if actual_output>0: #account for negative slopes 
			f = 1	
		else:
			f = 0;

		#update weights
		w_1 = w_now[0] + (D[3][idx]-f)*D[0][idx]
		w_2 = w_now[1] + (D[3][idx]-f)*D[1][idx]
		w_3 = w_now[2] + (D[3][idx]-f)*D[2][idx]
		w_now = np.array([w_1, w_2, w_3]);
		print(w_1,w_2,w_3)
		
		print(s)
		print(s+1)
		print(1/(s+1))
		print(D[3][idx]-f)
		print(f)

		#errors
		each_error.append((1/(s+1))*(D[3][idx]-f))
		
	###---  plot clusters of point data 
	plt.scatter(D[0,np.nonzero(D[3]>0)],D[1,np.nonzero(D[3]>0)],c='b')
	plt.scatter(D[0,np.nonzero(D[3]==0)],D[1,np.nonzero(D[3]==0)],c='r')

	#--- update x and y values of line determined by weights and plot
	yline = -(w_1*xline + w_3*1)/w_2	
	
	plt.ion()
	plt.plot(xline,yline,'k')
	plt.ylim(0,100)
	plt.xlim(0,100)
	plt.pause(.05)
	plt.cla()

	all_error = (1/(s+1))*np.sum(each_error)

	#--- exit code and plot final results once clusters are classified
	if all_error==0:  
		plt.ioff()
		plt.scatter(D[0,np.nonzero(D[3]>0)],D[1,np.nonzero(D[3]>0)],c='b')
		plt.scatter(D[0,np.nonzero(D[3]==0)],D[1,np.nonzero(D[3]==0)],c='r')
		plt.plot(xline,yline,'k')
		plt.ylim(0,100)
		plt.xlim(0,100)
		plt.show()
		break

