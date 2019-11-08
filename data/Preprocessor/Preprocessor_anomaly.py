from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
from pandas import DataFrame, Series
import tensorflow as tf
import matplotlib.pyplot as plt

ATTACK_TRAINING="data/KDDTraining.csv"


#1 : laod dataset
#trarining set

training_set = pd.read_csv(ATTACK_TRAINING, skipinitialspace=True,skiprows=1, usecols=[41], names=["attack_type"])
print(training_set)

training_set[training_set["attack_type"] !=  "normal"] = "anomaly"
print(training_set)

anomaly_oh  =pd.get_dummies(training_set.attack_type)
anomaly_oh.to_csv("data/anomaly_oh.csv")
print(anomaly_oh)