import sys
sys.path.append('/Users/zm/Desktop')
from numpy.linalg import inv
from scipy import misc
from random import uniform
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import math

#------------------------------
#-------Traning Data-----------
#------------------------------
def wifi_strength(distance):
	return (-30 - 1.3*distance + np.random.normal(0,sigma_n))
def wifi_strength_f(distance):
	return -30 - 1.3*distance

x1_data = np.array([])
x2_data = np.array([])
y_data = np.array([])
sigma_n = 2

#randomly generate some coordinates
for i in range(100):
	x_pos = uniform(0,38)
	y_pos = uniform(0,38)
	x1_data = np.append(x1_data,x_pos)
	x2_data = np.append(x2_data,y_pos)

#calculate signal intensities for all coordinates
for i in range(100):
	y_data = np.append(y_data,wifi_strength((x1_data[i]**2+x2_data[i]**2)**0.5))

#------------------------------
#--Hyperparamter Optimization--
#------------------------------

#function to calculate log likelihood based on hyperparameter
def log_likelihood_function(gph_l1, gph_l2,gph_sigma_f):
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
	m1 = np.dot(k_inv,y_data)
	part1 = np.dot(y_data.T,m1)
	part2 = math.log(np.linalg.det(k))
	return -0.5 * part1 - 0.5 * part2 - dim_x_data/2.0 * math.log(2*math.pi)

def partial_derivative(func, var=0, point=[]):
	args = point[:]
	def wraps(x):
		args[var] = x
		return func(*args)
	return misc.derivative(wraps,point[var],dx = 1e-6)

alpha = 0.1
nb_max_iter = 500
eps = 0.0001

gph_l1 = 17.8
gph_l2 = 17.8
gph_sigma_f = 2.86

ll_value = log_likelihood_function(gph_l1,gph_l2,gph_sigma_f)

cond = eps + 10.0
nb_iter = 0
tmp_ll_value = ll_value

#training the hyperparamter using gradient descent
while(cond>eps and nb_iter<nb_max_iter):
	tmp_gph_l1 = gph_l1 + \
	alpha * partial_derivative(log_likelihood_function,0,[gph_l1,gph_l2,gph_sigma_f])
	tmp_gph_l2 = gph_l2 + \
	alpha * partial_derivative(log_likelihood_function,1,[gph_l1,gph_l2,gph_sigma_f])
	tmp_gph_sigma_f = gph_sigma_f + \
	alpha * partial_derivative(log_likelihood_function,2,[gph_l1,gph_l2,gph_sigma_f])

	gph_l1 = tmp_gph_l1
	gph_l2 = tmp_gph_l2
	gph_sigma_f = tmp_gph_sigma_f

	ll_value = log_likelihood_function(gph_l1,gph_l2,gph_sigma_f)

	nb_iter = nb_iter + 1
	cond = abs(tmp_ll_value - ll_value)

	tmp_ll_value = ll_value

	print(ll_value)
print("Hyperparameters found using gradient descent: ")
print(gph_l1,gph_l2,gph_sigma_f)

#------------------------------
#---------Prediction-----------
#------------------------------
dim_x1_data = len(x1_data)
k = np.zeros((dim_x1_data,dim_x1_data))

#calculate the convariance matrix based on training data
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

x1 = np.arange(0,40,0.1)
x2 = np.arange(0,40,0.1)

dim_x1 = x1.shape[0]
dim_x2 = x2.shape[0]
dim_x = dim_x1 * dim_x2

Z = np.zeros((dim_x))
Z_var = np.zeros((dim_x))

i1_cp = 0
i2_cp = 0
sum_error = 0.0

#calculate every poing in the 40x40 space
for i in range(dim_x):
	x1_new = x1[i1_cp]
	x2_new = x2[i2_cp]
	k_new =np.zeros((dim_x1_data))
	for j in range(dim_x1_data):
		dx1 = x1_new - x1_data[j]
		dx2 = x2_new - x2_data[j]
		coef = -0.5 * (dx1**2/gph_l1**2 + dx2**2/gph_l2**2)
		k_new[j] = gph_sigma_f * gph_sigma_f * math.exp(coef)
	m1 = np.dot(k_new,k_inv)
	y_new = np.dot(m1,y_data)
	Z[i] = y_new
	sum_error += abs(y_new - wifi_strength_f((x1_new**2+x2_new**2)**0.5))
	Z_var[i] = k[0,0] - k_new.dot(k_inv.dot(np.transpose(k_new)))
	i2_cp += 1
	if(i2_cp == dim_x2):
		i1_cp += 1
		i2_cp = 0
	if(i1_cp == dim_x1):
		i1_cp = 0
print("Relative Prediction Error: "+str(sum_error/dim_x))

#------------------------------
#---------Ploting--------------
#------------------------------
Z = np.reshape(Z,(dim_x1,dim_x2))
Z_var = np.reshape(Z_var,(dim_x1,dim_x2))
x1,x2 = np.meshgrid(x1,x2)
fig = plt.figure()
ax = fig.gca(projection='3d')
#surf_mean = ax.plot_surface(x1,x2,Z,cmap=cm.coolwarm,linewidth=0, antialiased=False)
surf_var = ax.plot_surface(x1,x2,Z_var,cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.plot(x1_data,x2_data,y_data,linestyle="None", marker="o")
plt.show()








