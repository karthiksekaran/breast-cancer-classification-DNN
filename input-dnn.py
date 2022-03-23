import itertools
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn import metrics
from ggplot import *
import pandas as pd
from tkinter import *

CANCER_TRAINING = "train.csv"
CANCER_TEST = "test.csv"
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=CANCER_TRAINING,
    target_dtype=np.int,
    features_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=CANCER_TEST,
    target_dtype=np.int,
    features_dtype=np.int)
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20, 10],
                                            n_classes=2,
                                            model_dir="/Hell1")
prediction = classifier.fit(x=training_set.data, y=training_set.target, steps=2000).predict(test_set.data)
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score*100))

expected  = test_set.target
predicted = list(prediction)
confusion_matrix(expected,predicted)
print(expected)
print(predicted)
new_samples = np.array(
    [1, 1, 3, 1])
y = classifier.predict_proba(new_samples)
print ('Predictions: {}'.format(str(y)))