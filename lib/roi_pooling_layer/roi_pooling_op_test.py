import tensorflow as tf
import numpy as np
import roi_pooling_op
import roi_pooling_op_grad
import tensorflow as tf
import pdb


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

array = np.random.rand(32, 50, 50, 3)
"""array = np.zeros((32, 50, 50, 3))
for b in xrange(1,32):
	for x in xrange(1,50):
		for y in xrange(1,50):
			for c in xrange(1,3):
				array[b,x,y,c] = x+y"""
data = tf.convert_to_tensor(array, dtype=tf.float32)
rois_array = [[0, 10, 10, 20, 20], [31, 30, 30, 40, 40]]
rois = tf.convert_to_tensor(rois_array, dtype=tf.float32)

W = weight_variable([3, 3, 3, 2])
h = conv2d(data, W)
proposal1 = h[:,10:20,10:20,:]
proposal2 = h[:,30:40,30:40,:]
print proposal1, proposal2

[y, argmax_x, argmax_y] = roi_pooling_op.roi_pool(h, rois, 6, 6, 1.0/3)
#pdb.set_trace()
y_data = tf.convert_to_tensor(np.ones((2, 6, 6, 2)), dtype=tf.float32)
print y, argmax_x, argmax_y

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
#pdb.set_trace()
for step in xrange(10):
    sess.run(train)
    print(step, sess.run(W))
    print(sess.run(y))

#with tf.device('/gpu:0'):
#  result = module.roi_pool(data, rois, 1, 1, 1.0/1)
#  print result.eval()
#with tf.device('/cpu:0'):
#  run(init)
