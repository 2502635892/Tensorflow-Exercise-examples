from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# Make up some real data
# training data
mu1 = np.array([20,20])
Sigma1 = np.array([[15,0.5],[0.5,15]])

x_data_train1 = np.random.multivariate_normal(mu1,Sigma1,1000)
y_data_train1 = np.array([[1,0]]*1000)

mu2 = np.array([50,50])
Sigma2 = np.array([[15,0.5],[0.5,15]])

x_data_train2 = np.random.multivariate_normal(mu2,Sigma2,1000)
y_data_train2 = np.array([[0,1]]*1000)

#print ( x_data_train1.shape,y_data_train1.shape)

x_data_train = np.vstack((x_data_train1,x_data_train2))
y_data_train = np.vstack((y_data_train1,y_data_train2))
#print (y_data_train)
#print (y_data_train.shape)

## define placeholder for inputs && outputs to network
xs = tf.placeholder(tf.float32,[None,2])
ys = tf.placeholder(tf.float32,[None,2])

##add 4 hidden layer to network

l1 = add_layer(xs,2, 8 , activation_function=tf.nn.relu)
l2 = add_layer(l1,8, 16, activation_function=tf.nn.sigmoid)
l3 = add_layer(l2,16,32,activation_function=tf.nn.relu)
l4 = add_layer(l3,32,5,activation_function=tf.nn.sigmoid)
## prediction the outputs
prediction = add_layer(l4,5,2,activation_function = tf.nn.softmax)

## define loss function
loss = tf.reduce_mean( tf.reduce_sum( tf.square( ys - prediction),reduction_indices=[1] ) )
#define training action
#train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
##initialize the variable we ever defined
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(8001):
    sess.run( train_step , feed_dict = {xs:x_data_train,ys:y_data_train} )
    if i%50 == 0:
        print (i, sess.run( loss,feed_dict = {xs:x_data_train,ys:y_data_train} ) )


# testing data

x_data_testing1 = np.random.multivariate_normal(mu1,Sigma1,500)
y_data_testing1 = np.array([[1,0]]*500)

x_data_testing2 = np.random.multivariate_normal(mu2,Sigma2,500)
y_data_testing2 = np.array([[0,1]]*500)

x_data_tesing = np.vstack((x_data_testing1,x_data_testing2))
y_data_testing = np.vstack((y_data_testing1,y_data_testing2))

pre = sess.run( prediction,feed_dict={xs:x_data_tesing } )
print(pre)

predic=np.array([[0,0]]*1000)
for i in range(1000):
    if pre[i,0] == np.max(pre[i,:]):
        predic[i,0] = 1
    else:
        predic[i,1] = 1

print(predic)
Err = predic - y_data_testing
count = 0
for i in range(1000):
    if Err[i,0]==0 and Err[i,1]==0 :
        count=count+1

print('the prediction correct rate :%f'%( count/1000.0))

plt.plot(x_data_train[:,0],x_data_train[:,1],'b*')
plt.show()

plt.plot(x_data_tesing[:,0],x_data_tesing[:,1],'r*')
plt.show()
