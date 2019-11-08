# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

PATH="tmp/anomaly_detection"
TB_SUMMARY_DIR = PATH + "DNN_0.1_H4"
LEARNINGRATE=0.1
GLOABAL_STEP=2000

# Training malware type based on various features
train_x = np.loadtxt('training.csv', delimiter=',',skiprows=1, dtype=np.float32)
train_y = np.loadtxt('anomaly_oh.csv', delimiter=',',skiprows=1, dtype=np.float32)
train_x_data = train_x[1:100000, 1:]
train_y_data = train_y[1:100000, 1:]
#print(train_x_data)

#Testing malware type based on various features
test_x = np.loadtxt('training.csv', delimiter=',',skiprows=1, dtype=np.float32)
test_y = np.loadtxt('anomaly_oh.csv', delimiter=',',skiprows=1, dtype=np.float32)
test_x_data = train_x[100001:, 1:]
test_y_data = train_y[100001:, 1:]

#print(train_x_data, train_y_data)
#print(train_x_data.shape, train_y_data.shape)


nb_classes = 2  #number of attacks

X = tf.placeholder(tf.float32, [None, 8])
Y = tf.placeholder(tf.float32, [None, nb_classes])


with tf.name_scope("hidden_layer1") as scope:

    W1 = tf.Variable(tf.random_normal([8, 16]), name='weight1')
    b1 = tf.Variable(tf.random_normal([16]), name='bias1')

    L1 = tf.add(tf.matmul(X, W1), b1)
    L1 = tf.nn.relu(L1)

    ###########tensorboard log###########
    W1_hist = tf.summary.histogram("weights1", W1)
    b_hist = tf.summary.histogram("bias1", b1)
    L1_hist = tf.summary.histogram("Layer1", L1)
    ###########tensorboard log###########


with tf.name_scope("hidden_layer2") as scope:
    W2 = tf.Variable(tf.random_normal([16, 32]), name='weight2')
    b2 = tf.Variable(tf.random_normal([32]), name='bias2')

    L2 = tf.add(tf.matmul(L1, W2), b2)
    L2 = tf.nn.relu(L2)

    ###########tensorboard log###########
    W2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("bias2", b2)
    L2_hist = tf.summary.histogram("Layer2", L2)
    ###########tensorboard log###########

with tf.name_scope("hidden_layer3") as scope:
    W3 = tf.Variable(tf.random_normal([32, 16]), name='weight3')
    b3 = tf.Variable(tf.random_normal([16]), name='bias3')

    L3 = tf.add(tf.matmul(L2, W3), b3)
    L3 = tf.nn.relu(L3)

    ###########tensorboard log###########
    W3_hist = tf.summary.histogram("weights3", W3)
    b3_hist = tf.summary.histogram("bias3", b3)
    L3_hist = tf.summary.histogram("Layer3", L3)
    ###########tensorboard log###########

with tf.name_scope("hidden_layer4") as scope:
    W4 = tf.Variable(tf.random_normal([16, nb_classes]), name='weight4')
    b4 = tf.Variable(tf.random_normal([nb_classes]), name='bias4')

    hypothesis=tf.add(tf.matmul(L3, W4), b4)
    ###########tensorboard log###########
    W4_hist = tf.summary.histogram("weights4", W4)
    b4_hist = tf.summary.histogram("bias4", b4)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)
    ###########tensorboard log###########

with tf.name_scope("xent"):
    cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=hypothesis))
    cost_scalar = tf.summary.scalar("cost", cost)

with tf.name_scope("prediction"):
    prediction = tf.argmax(hypothesis, 1)
    target=tf.argmax(Y,1)

with tf.name_scope("accracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(prediction, target) #is equal?

    with tf.name_scope("accracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy_scalar = tf.summary.scalar("accuracy", accuracy)

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNINGRATE).minimize(cost)


#inintialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Merge all the summary
summary =tf.summary.merge_all()

#Create summary writer
writer =tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph) #Add graph in the tensorboard
###############summary##############


# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(GLOABAL_STEP+1):
        #summary with optimizer
        s, _ = sess.run([summary, optimizer], feed_dict={X: train_x_data, Y:  train_y_data})
        writer.add_summary(s, step)

        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                X: train_x_data, Y: train_y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))


# Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: test_x_data})
    targ = sess.run(target, feed_dict={Y: test_y_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, targ):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


