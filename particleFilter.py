import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from random import uniform
from numpy.linalg import inv
import math
import scipy.stats


#Define the Person's location
class Person:
	def __init__(self,vertex1,vertex2,position,graph):
		self.vertex1 = vertex1
		self.vertex2 = vertex2
		self.position = position
		self.graph = graph
	def coordinates(self):
		graph = self.graph
		x1 = graph.node[self.vertex1]["pos"][0]
		y1 = graph.node[self.vertex1]["pos"][1]
		x2 = graph.node[self.vertex2]["pos"][0]
		y2 = graph.node[self.vertex2]["pos"][1]
		maxLength = G.get_edge_data(self.vertex1,self.vertex2)["length"]
		currX = x1 + (x2-x1)*(self.position/maxLength)
		currY = y1 + (y2-y1)*(self.position/maxLength)
		return (currX,currY)

class Particle:
	def __init__(self,vertex1,vertex2,position,weight,graph):
		self.vertex1 = vertex1
		self.vertex2 = vertex2
		self.position = position
		self.weight = weight
		self.graph = graph
	def coordinates(self):
		graph = self.graph
		x1 = graph.node[self.vertex1]["pos"][0]
		y1 = graph.node[self.vertex1]["pos"][1]
		x2 = graph.node[self.vertex2]["pos"][0]
		y2 = graph.node[self.vertex2]["pos"][1]
		maxLength = G.get_edge_data(self.vertex1,self.vertex2)["length"]
		currX = x1 + (x2-x1)*(self.position/maxLength)
		currY = y1 + (y2-y1)*(self.position/maxLength)
		return (currX,currY)

#Set-up
plt.gca().set_aspect("equal", adjustable="box")
G=nx.Graph()

velocity_mean = 1.4
velocity_variance = 0.5
time = 105

#Construct the map
G.add_node(0,pos=(-10,0))
G.add_node(1,pos=(0,0))
G.add_node(2,pos=(10,0))
G.add_node(3,pos=(2,6))
G.add_node(4,pos=(-6,-6))
G.add_node("P",pos=(-3,0),color="blue")

G.add_edge(0,1,length=10.0)
G.add_edge(1,2,length=10.0)
G.add_edge(1,3,length=6.5)
G.add_edge(1,4,length=8.5)

#motion model
def move(person):
	distance = np.random.normal(velocity_mean,velocity_variance)
	direction = random.randint(0,1)
	lowLimit = 0
	UpLimit = G.get_edge_data(person.vertex1,person.vertex2)["length"]
	position = person.position
	
	#moving
	if(direction==0):
		position -= distance
		if(position>=0):
			person.position = position
			G.node["P"]["pos"] = person.coordinates()
		else:
			#switch to another edge or turn around
			neighbors = [n for n in iter(G[person.vertex1])]
			if(len(neighbors)!=0):
				randIndex = random.randint(0,len(neighbors)-1)
				dest = neighbors[randIndex]
				position = -position
				person.vertex2 = dest
				person.position = position
				G.node["P"]["pos"] = person.coordinates()
			else:
				position = -position
				person.position = position
				G.node["P"]["pos"] = person.coordinates()

	else:
		position += distance
		if(position<=UpLimit):
			#update graph
			person.position = position
			G.node["P"]["pos"] = person.coordinates()
		else:
			#switch to another edge or turn around
			neighbors = [n for n in iter(G[person.vertex2])]
			if(len(neighbors)!=0):
				randIndex = random.randint(0,len(neighbors)-1)
				dest = neighbors[randIndex]
				position -= UpLimit
				person.vertex1 = person.vertex2
				person.vertex2 = dest
				person.position = position
				G.node["P"]["pos"] = person.coordinates()
			else:
				position -= position
				position = UpLimit - position
				person.position = position
				G.node["P"]["pos"] = person.coordinates()

#motion model
def moveParticle(person):
	distance = np.random.normal(velocity_mean,velocity_variance)
	direction = random.randint(0,1)
	lowLimit = 0
	UpLimit = G.get_edge_data(person.vertex1,person.vertex2)["length"]
	position = person.position
	
	#moving
	if(direction==0):
		position -= distance
		if(position>=0):
			person.position = position
		else:
			#switch to another edge or turn around
			neighbors = [n for n in iter(G[person.vertex1])]
			if(len(neighbors)!=0):
				randIndex = random.randint(0,len(neighbors)-1)
				dest = neighbors[randIndex]
				position = -position
				person.vertex2 = dest
				person.position = position
			else:
				position = -position
				person.position = position

	else:
		position += distance
		if(position<=UpLimit):
			#update graph
			person.position = position
		else:
			#switch to another edge or turn around
			neighbors = [n for n in iter(G[person.vertex2])]
			if(len(neighbors)!=0):
				randIndex = random.randint(0,len(neighbors)-1)
				dest = neighbors[randIndex]
				position -= UpLimit
				person.vertex1 = person.vertex2
				person.vertex2 = dest
				person.position = position
			else:
				position -= position
				position = UpLimit - position
				person.position = position



#Run the Gaussian Process Model
#------------------------------
#--------Access Point----------
#------------------------------
access_x1 = [-10,0,10,0,0]
access_x2 = [0,0,0,6,-6]


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
	x_pos = uniform(-10,10)
	y_pos = uniform(-6,6)
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

def predict(x1_new,x2_new,true_y):
	k_new =np.zeros((dim_x1_data))
	for j in range(dim_x1_data):
		dx1 = x1_new - x1_data[j]
		dx2 = x2_new - x2_data[j]
		coef = -0.5 * (dx1**2/gph_l1**2 + dx2**2/gph_l2**2)
		k_new[j] = gph_sigma_f * gph_sigma_f * math.exp(coef)
	m1 = np.dot(k_new,k_inv)
	u = []
	std = []
	for i in range(len(y_data)):
		u.append(np.dot(m1,y_data[i]))
		std.append(math.sqrt(k[0,0] - k_new.dot(k_inv.dot(np.transpose(k_new)))))
	#print("Prediction Value:",y_new)
	y_sum = 1
	for i in range(len(u)):
		y_sum *= scipy.stats.norm(u[i],std[i]).pdf(true_y[i])
	y_sum = y_sum**(1.0/len(y_data))
	return y_sum

#------------------------------
#---------Simulation-----------
#------------------------------
p = Person(0,1,7,G)

observe_y = []
for z in range(len(access_x1)):
	observe_y.append(wifi_strength(distance_between(p.coordinates()[0],access_x1[z],p.coordinates()[1],access_x2[z])))
particles = []
for edge in G.edges(data=True):
	v1 = edge[0]
	v2 = edge[1]
	UpLimit = edge[2]["length"]
	for i in np.arange(0.0,UpLimit,0.1):
		particle = Particle(v1,v2,i,0,G)
		point_weight = predict(particle.coordinates()[0],particle.coordinates()[1],observe_y)
		particle.weight = point_weight
		particles.append(particle)

#draw the graph
def drawMap():
	color_map = []
	for node in G:
		if node == "P":
			color_map.append("blue")
		else:
			color_map.append("red")

	pos=nx.get_node_attributes(G,"pos")
	nx.draw(G,pos,node_color=color_map,with_labels=True)
	for particle in particles:
		plt.plot(particle.coordinates()[0],particle.coordinates()[1],'ro',color='green')
	plt.show()

#drawMap()
#start moving
error_sum = 0
error_count = 0
for i in range(time):
	#print(G.node["P"]["pos"])
	#Resampling
	probability = []
	for j in range(len(particles)):
		probability.append(particles[j].weight) 
	probs = np.array(probability)
	probs /= probs.sum()
	resampled_index = np.random.choice(len(particles),len(particles),p=probs)
	resampled_particles = []
	for j in range(len(resampled_index)):
		chosenP = particles[resampled_index[j]]
		resampled_particles.append(Particle(chosenP.vertex1,chosenP.vertex2,chosenP.position,chosenP.weight,G))
	particles = resampled_particles
	#draw new xt
	#drawMap()
	move(p)
	observe_y = []
	for z in range(len(access_x1)):
		observe_y.append(wifi_strength(distance_between(p.coordinates()[0],access_x1[z],p.coordinates()[1],access_x2[z])))
	optimal_particle = None
	for particle in particles:
		moveParticle(particle)
		point_weight = particle.weight * predict(particle.coordinates()[0],particle.coordinates()[1],observe_y)
		particle.weight = point_weight
		if(optimal_particle is None):
			optimal_particle = particle
		elif(optimal_particle.weight < point_weight):
			optimal_particle = particle
	if(i>=6):
		error_sub = distance_between(p.coordinates()[0],optimal_particle.coordinates()[0],p.coordinates()[1],optimal_particle.coordinates()[1])
		print(error_sub)
		error_sum += error_sub
		error_count += 1
print("Average Tracking Error: ")
print(error_sum/error_count)



