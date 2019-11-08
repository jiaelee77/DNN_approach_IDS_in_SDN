'''
load training set % test set
i used only divide training set
training set : 1-100000 raws
test set : 100001-the rest(about 25000)raws

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
from pandas import DataFrame, Series
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

tf.logging.set_verbosity(tf.logging.INFO)


ATTACK_TRAINING="KDDTraining.csv"
ATTACK_TEST="KDDTest.csv"
ATTACK_PREDICTION="KDDPrediction.csv"

COLUMNS = ["duration", "protocol_type", "src_bytes", "dst_bytes", "count", "srv_count", "attack_type"]
FEATURES = ["duration", "protocol_type", "src_bytes", "dst_bytes", "count", "srv_count"]
LABEL = "attack"


#1 : laod dataset
#trarining set
training_set1 = pd.read_csv(ATTACK_TRAINING, skipinitialspace=True, skiprows=1, usecols=[0,4,5,22,23],
                             names=["duration", "src_bytes", "dst_bytes", "count", "srv_count"])
training_set2 = pd.read_csv(ATTACK_TRAINING, skipinitialspace=True, skiprows=1, usecols=[1],
                              names=["protocol_type"])
training_set3 = pd.read_csv(ATTACK_TRAINING, skipinitialspace=True, skiprows=1, usecols=[41],
                              names=["attack_type"])
'''#test set
test_set1 = pd.read_csv(ATTACK_TEST, skipinitialspace=True, skiprows=1, usecols=[0,4,5,22,23],
                             names=["duration", "src_bytes", "dst_bytes", "count", "srv_count"])
test_set2 = pd.read_csv(ATTACK_TEST, skipinitialspace=True, skiprows=1, usecols=[1],
                              names=["protocol_type"])
test_set3 = pd.read_csv(ATTACK_TEST, skipinitialspace=True, skiprows=1, usecols=[41],
                              names=["attack_type"])

#prediction set
prediction_set1 = pd.read_csv(ATTACK_PREDICTION, skipinitialspace=True, skiprows=1, usecols=[0,4,5,22,23],
                             names=["duration", "src_bytes", "dst_bytes", "count", "srv_count"])
prediction_set2 = pd.read_csv(ATTACK_PREDICTION, skipinitialspace=True, skiprows=1, usecols=[1],
                              names=["protocol_type"])
prediction_set3 = pd.read_csv(ATTACK_PREDICTION, skipinitialspace=True, skiprows=1, usecols=[41],
                              names=["attack_type"])

'''


#2: extract set
#training
preprocessed_training=pd.concat([training_set1, training_set2,training_set3], axis=1)
preprocessed_training.to_csv('preprocessed_training.csv')
#print(preprocessed_training.duration.max())

'''
#test set
preprocessed_test=pd.concat([test_set1, test_set2,test_set3], axis=1)
preprocessed_test.to_csv('data/preprocessed_test.csv')

#prediction set
preprocessed_prediction=pd.concat([prediction_set1, prediction_set2], axis=1)
preprocessed_prediction.to_csv('data/preprocessed_prediction.csv')
print(preprocessed_prediction.duration.max())

'''


#3 : one-hot encoding
#trarining set
training_oh_protocol= pd.get_dummies(training_set2.protocol_type)
training_oh_attack_type=pd.get_dummies(training_set3.attack_type)

'''
#test set
test_oh_protocol= pd.get_dummies(test_set2.protocol_type)
test_oh_attack_type=pd.get_dummies(test_set3.attack_type)

#prediction set
prediction_oh_protocol= pd.get_dummies(prediction_set2.protocol_type)
prediction_oh_attack_type=pd.get_dummies(prediction_set3.attack_type)
'''

#4: normalization
scaler=MinMaxScaler()

scaled_training = pd.DataFrame(scaler.fit_transform(training_set1),columns=training_set1.columns)
#scaled_test = pd.DataFrame(scaler.fit_transform(test_set1),columns=test_set1.columns)
#scaled_prediction = pd.DataFrame(scaler.fit_transform(prediction_set1),columns=prediction_set1.columns)


#5: result set
training=pd.concat([scaled_training, training_oh_protocol], axis=1)
training.to_csv('training.csv')
training_set3.to_csv('training_attack.csv')



#6 : display


#7 : check attck
training_oh_attack_type.to_csv('oh_training_attack.csv')
#test_oh_attack_type.to_csv('oh_test_attack.csv')
#prediction_oh_attack_type.to_csv('oh_prediction_attack.csv')

col1=training_set1.duration

plt.show()
