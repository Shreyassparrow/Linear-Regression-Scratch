#formula of linear regression > y = wx+b > gradient of the slope (W) and bias (b) through multiple iterations.
#learning rate 0.1 has more impact then other
import numpy as np
import tensorflow as tf
import pandas as pd


'''def generate_dataset():
	x_batch = [2,4,6,8,0,1,3,5,7,9,10]
	#we want numpy to generate 100 points with value between 0 and 2, spreaded evenly. The result is a numpy array . shape method give the dimension
	#print(x_batch)
	y_batch = [7,11,15,19,3,5,9,13,17,21,23]
	#numpy.random.randn = creates an array of specified shape and fills it with random values as per standard normal distribution.
	#randomly generate y such that it has a gradient of 1.5 (W) and some form of randomness using np.random.randn(). To make things interesting, we set y-intercept b to 0.5.
	#print(y_batch)
	return x_batch,y_batch
#generate_dataset()




#LInear regression
def linear_regression():
	#Declaring x and y as placeholders mean that we need to pass in values at a later time

	#In the first argument of tf.placeholder, we define the data type as float32 a common data type in placeholder. The second argument is the shape of the placeholder set to None as we want it to be determined during training time. The third argument lets us set the name for the placeholder.

	x = tf.placeholder(tf.float32, shape=(None, ), name='x')
  	y = tf.placeholder(tf.float32, shape=(None, ), name='y')
	#To elaborate, it is a mechanism in TensorFlow that allows variables to be shared in different parts of the graph without passing references to the variable around. Note that even though we do not reuse variables here, it is a good practice to name them appropriately.
	with tf.variable_scope('lreg') as scope:
    		w = tf.Variable(np.random.normal(), name='W')
    		b = tf.Variable(np.random.normal(), name='b')
		y_pred = tf.add(tf.multiply(w, x), b)
		loss = tf.reduce_mean(tf.square(y_pred - y))
		print(w)
	return x, y, y_pred, loss




#start the training
def run():
	x_batch, y_batch = generate_dataset()
	print("x:",x_batch,"y:",y_batch)
	x, y, y_pred, loss = linear_regression()
	#x = placeholder
	#y = placeholder
	#y_pred = placeholder
	#loss= placeholder
	
	train_step = tf.compat.v1.train.AdamOptimizer().minimize(loss)

	#Now, all thats left is to train it:
	sess = tf.compat.v1.Session()
	#print(sess)
	init = tf.compat.v1.global_variables_initializer()
	#print(init)
	sess.run(init) #make sure you do this!
	
	#more epochs = longer training time.
	for i in range(1000):
		sess.run(train_step,feed_dict = {x: x_batch, y: y_batch})
		print(i, "loss:",sess.run(loss,feed_dict = {x: x_batch, y: y_batch}))
	print('Predicting')
    	y_pred_batch = sess.run(y_pred, {x : x_batch})
	print(y_pred_batch)
	print(y_batch)
	
	
	

if __name__ == "__main__":
	run()'''
	


#setting the datas and showing in a clear form
data = pd.read_csv('Dataset/shreyasData.csv')
#print("Data Shape:", data.shape) 
#print(data.head())



# Feature Matrix 
x_orig = data.iloc[:,:-1].values
print("x_orig : ",x_orig)
x_orig = x_orig[:]
#print(x_orig)


# Data labels 
y_orig = data.iloc[:, -1:].values
#print y_orig
y_orig = y_orig[:]
#print(y_orig)




x = x_orig
#print(x)
y = y_orig
#print(x,y)
_,m = x.shape
#print(_)
_,n = y.shape
#print(_)
print(m,n)


#hyperparameters
#learning_rate = 0.01
#slope * x + noise

X = tf.placeholder(tf.float64, shape=[None,m], name="Input")
Y = tf.placeholder(tf.float64, shape=[None,n], name="Output")


W = tf.Variable(np.random.randn(m,n)*np.sqrt(1/(m+n)),name = "weights1")
#W = tf.Variable(W*np.sqrt(2/(m+n)))

B = tf.Variable(np.random.randn(n),name = "bias")
#B = tf.Variable(B*np.sqrt(2/(m+n)))

pred =tf.add(tf.matmul(X,W),B)

cost = tf.sqrt(tf.reduce_mean(tf.square(pred - Y))) #tf.reduce_mean(-tf.reduce_sum(Y * tf.math.log(pred), reduction_indices=[1]))
# #formula 1/2nE(pred(i)-Y(i))^2 = loss

optimizer = tf.compat.v1.train.AdamOptimizer(0.1).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
epoch = 0
c = 0
while(epoch<10):
	epoch+=1
	sess.run(optimizer,feed_dict = {X:x,Y:y})
	xtrain = sess.run(X,feed_dict = {X:x}) 
	predY = sess.run(pred,feed_dict = {X:x})
	w1 = sess.run(W)
	b = sess.run(B)
	#print(len(predY))
	#print(xtrain.shape,w1.shape)
	#print y
	if not epoch%1:
		
		c = sess.run(cost,feed_dict = {X:x,Y:y})
		w1 = sess.run(W)
		b = sess.run(B)
		
		if round(c,2) == 0.0:
			break
		print("epoch = ",epoch,"cost = ",c,"Weights1 = ",w1,"Bias = ",b)
	
#print(trainY)
ypred = sess.run(pred,feed_dict = {X:x})
for i in ypred:
	#print(i)
	pass
 
 
