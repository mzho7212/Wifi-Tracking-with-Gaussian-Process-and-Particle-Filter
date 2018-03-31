from numpy.linalg import inv
from scipy import misc
from random import uniform
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import math

#------------------------------
#--------Access Point----------
#------------------------------
access_x1 = [0,20,40,0,40]
access_x2 = [0,20,40,40,0]


#------------------------------
#-------Traning Data-----------
#------------------------------
num_data = 100
def distance_between(x1,x2,y1,y2):
	return ((x1-x2)**2 + (y1-y2)**2)**0.5
def wifi_strength(distance):
	return (-30 - 1.3*distance + np.random.normal(0,sigma_n))
def wifi_strength_f(distance):
	return -30 - 1.3*distance

x1_data = np.array([])
x2_data = np.array([])
y_data = []
sigma_n = 2

#randomly generate some coordinates
for i in range(num_data):
	x_pos = uniform(0,38)
	y_pos = uniform(0,38)
	x1_data = np.append(x1_data,x_pos)
	x2_data = np.append(x2_data,y_pos)

#calculate signal intensities for all access point and all coordinates
for i in range(len(access_x1)):
	x1_access = access_x1[i]
	x2_access = access_x2[i]
	y_curr_data = np.array([])
	for j in range(num_data):
		distance = distance_between(x1_access,x1_data[j],x2_access,x2_data[j])
		y_curr_data = np.append(y_curr_data,wifi_strength(distance))
	y_data.append(y_curr_data)

#------------------------------
#--Hyperparamter Optimization--
#------------------------------

#function to calculate log likelihood based on hyperparameter
def log_likelihood_function(gph_l1, gph_l2,gph_sigma_f,y_index):
	dim_x_data = len(x1_data)
	k = np.zeros((dim_x_data,dim_x_data))
	for i in range(dim_x_data):
		for j in range(i+1):
			dx1 = x1_data[i] - x1_data[j]
			dx2 = x2_data[i] - x2_data[j]
			coef = -0.5 * (dx1**2/gph_l1**2 + dx2**2/gph_l2**2)
			k[i,j] = gph_sigma_f * gph_sigma_f * math.exp(coef)
			k[j,i] = k[i,j]
			if i == j:
				k[i,j] += sigma_n * sigma_n
	k_inv = inv(k)
	m1 = np.dot(k_inv,y_data[y_index])
	part1 = np.dot(y_data[y_index].T,m1)
	part2 = math.log(np.linalg.det(k))
	return -0.5 * part1 - 0.5 * part2 - dim_x_data/2.0 * math.log(2*math.pi)

alpha = 0.00000003
nb_max_iter = 500
eps = 0.001

gph_l1_data = [17.8,17.8,17.8,17.8,17.8]
gph_l2_data = [17.8,17.8,17.8,17.8,17.8]
gph_sigma_f_data = [2.86,2.86,2.86,2.86,2.86]

#generate a gaussian process for each access point
for index in range(len(y_data)):
	print("Access Point",index+1,"Training: ")

	gph_l1 = gph_l1_data[index]
	gph_l2 = gph_l2_data[index]
	gph_sigma_f = gph_sigma_f_data[index]

	ll_value = log_likelihood_function(gph_l1,gph_l2,gph_sigma_f,index)

	cond = eps + 10.0
	nb_iter = 0
	tmp_ll_value = ll_value
	print(ll_value)

	#training the hyperparamter using gradient descent
	while(cond>eps and nb_iter<nb_max_iter):
		dim_x1_data = len(x1_data)
		K = np.zeros((dim_x1_data,dim_x1_data))
		A = np.zeros((dim_x1_data,dim_x1_data))
		B = np.zeros((dim_x1_data,dim_x1_data))
		C = np.zeros((dim_x1_data,dim_x1_data))
		for i in range(dim_x1_data):
			for j in range(i+1):
				dx1 = x1_data[i] - x1_data[j]
				dx2 = x2_data[i] - x2_data[j]
				coef = (dx1**2/gph_l1**2 + dx2**2/gph_l2**2)
				K[i,j] = gph_sigma_f * gph_sigma_f * math.exp(-0.5 *coef)
				K[j,i] = K[i,j]
				A[i,j] = 2 * gph_sigma_f * math.exp(-0.5 *coef)
				A[j,i] = A[i,j]
				B[i,j] = K[i,j] * coef / gph_l1
				B[j,i] = B[i,j]
				C[i,j] = K[i,j] * coef / gph_l2
				C[j,i] = C[i,j]
				if(i==j):
					K[i,j] += sigma_n * sigma_n
		Kinv = np.linalg.inv(K)
		KinvY = Kinv.dot(y_data[index])
		#lsigma_f = 0.5*np.trace(Kinv.dot(A)) - 0.5*(np.transpose(y).dot(Kinv).dot(A).dot(Kinv).dot(y)).squeeze()
		lsigma_f = 0.5*np.trace(np.subtract(KinvY.dot(KinvY.T),Kinv).dot(A))
		#ll = 0.5*np.trace(Kinv.dot(B)) - 0.5*(np.transpose(y).dot(Kinv).dot(B).dot(Kinv).dot(y)).squeeze()
		ll1 = 0.5*np.trace(np.subtract(KinvY.dot(KinvY.T),Kinv).dot(B))
		ll2 = 0.5*np.trace(np.subtract(KinvY.dot(KinvY.T),Kinv).dot(C))
		#lsigma_n = 0.5*np.trace(Kinv.dot(np.eye(len(X)))) - 0.5*(np.transpose(y).dot(Kinv).dot(np.eye(len(X))).dot(Kinv).dot(y)).squeeze()
		#lsigma_n = 0.5*np.trace(np.subtract(KinvY.dot(KinvY.T),Kinv).dot(np.multiply(np.eye(dim_x1_data),2*x[3])))
		
		tmp_gph_l1 = gph_l1 + \
		alpha * ll1
		tmp_gph_l2 = gph_l2 + \
		alpha * ll2
		tmp_gph_sigma_f = gph_sigma_f + \
		alpha * lsigma_f

		gph_l1 = tmp_gph_l1
		gph_l2 = tmp_gph_l2
		gph_sigma_f = tmp_gph_sigma_f

		ll_value = log_likelihood_function(gph_l1,gph_l2,gph_sigma_f,index)

		nb_iter = nb_iter + 1
		cond = abs(tmp_ll_value - ll_value)

		tmp_ll_value = ll_value

		print(tmp_ll_value)

	gph_l1_data[index] = gph_l1
	gph_l2_data[index] = gph_l2
	gph_sigma_f_data[index] = gph_sigma_f
	print("Hyperparameters found using gradient descent: ")
	print(gph_l1,gph_l2,gph_sigma_f)


#------------------------------
#---------Prediction-----------
#------------------------------
def predict(x1_new,x2_new):
	dim_x1_data = len(x1_data)
	k = np.zeros((dim_x1_data,dim_x1_data))
	for i in range(dim_x1_data):
		for j in range(i+1):
			dx1 = x1_data[i] - x1_data[j]
			dx2 = x2_data[i] - x2_data[j]
			coef = -0.5 * (dx1**2/gph_l1**2 + dx2**2/gph_l2**2)
			k[i,j] = gph_sigma_f * gph_sigma_f * math.exp(coef)
			k[j,i] = k[i,j]
			if(i==j):
				k[i,j] += sigma_n * sigma_n
	k_inv = inv(k)

	k_new =np.zeros((dim_x1_data))
	for j in range(dim_x1_data):
		dx1 = x1_new - x1_data[j]
		dx2 = x2_new - x2_data[j]
		coef = -0.5 * (dx1**2/gph_l1**2 + dx2**2/gph_l2**2)
		k_new[j] = gph_sigma_f * gph_sigma_f * math.exp(coef)
	m1 = np.dot(k_new,k_inv)
	y_new = []
	for i in range(len(y_data)):
		y_new.append(np.dot(m1,y_data[i])+120)
	print("Prediction Value:",y_new)
	return y_new

#------------------------------
#-----------Ploting------------
#------------------------------
for i in range(5):
	x1_new = uniform(0,38)
	x2_new = uniform(0,38)
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	zpos = [-120,-120,-120,-120,-120]
	dx = np.ones(10)
	dy = np.ones(10)
	ax.bar3d(access_x1,access_x2,zpos,dx,dy,predict(x1_new,x2_new),color="#00ceaa")
	ax.plot([x1_new],[x2_new],[-120],linestyle="None", marker="o")
	plt.show()
