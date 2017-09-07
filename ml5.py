import tensorflow as tf
import numpy as np
#import tensorflow.examples.tutorials.mnist.input_data as input_data
#mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)


n_nodes_hl1=3000
n_nodes_hl2=1500
n_nodes_hl3=1500
n_classes=1
x=tf.placeholder('float',[None,1])
y=tf.placeholder('float',[None,1])

def neural_network_model(data):
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([1,n_nodes_hl1],stddev=0.01)),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2],stddev=0.01)),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3],stddev=0.01)),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_classes],stddev=0.01)),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1=tf.add(tf.matmul(data, hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1=tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output=tf.matmul(l3, output_layer['weights'])+output_layer['biases']

    return output

print ("Hello")

def train_neural_network(x):
    prediction=neural_network_model(x)
    cost = tf.reduce_mean(tf.squared_difference(prediction, y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        cap = 50
        for epoch in range(2000):
            sx = np.random.randint(cap, size=(100, 1))
            # sx = np.random.rand(100,1)
            sy = np.sqrt(sx)
            op, c = sess.run([optimizer, cost], feed_dict={x: sx, y: sy})
            if epoch % 100 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "%.03f" % c)

        # sx = np.random.rand(10,1)
        sx = np.random.randint(cap, size=(10, 1))
        sy = np.sqrt(sx)
        print "Input"
        print sx
        print "Expected Output"
        print sy
        print "Predicted Output"
        print sess.run(prediction, feed_dict={x: sx, y: sy})
        print "Error"
        print sess.run(cost, feed_dict={x: sx, y: sy})



train_neural_network(x)
