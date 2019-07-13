# -*- coding: utf-8 -*-
#author: guifeng tang
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import tensorflow as tf

import numpy as np
from numpy import array
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def GetSequences(f):
    seqslst = []
    while True:
         s = f.readline()
         if not s:
             break
         else:
             if '>' not in s:
                seq = s.split('\n')[0]
                seqslst.append(seq)
    return seqslst

def ConstructPartitionOfSet(y, folds_num, seed):
    folds = []
    kf = KFold(n_splits=folds_num, shuffle=True, random_state=np.random.RandomState(seed))
    for train_index, test_index in kf.split(y):
        folds.append((train_index, test_index))
    return folds

def GetPartitionOfSamples(X, y, train_index, test_index):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    return X_train, X_test, y_train

def MakePrediction(X_train, X_test, y_train):
    classifier = clf.fit(X_train, y_train)
    predict_test_proba = classifier.predict_proba(X_test)[:, 1]
    return predict_test_proba

def EvaluatePerformances(real_labels, predicted_probas):
    precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probas, pos_label=1)
    aupr_final = auc(recall, precision)
    fpr, tpr, auc_thresholds = roc_curve(real_labels, predicted_probas)
    auc_final = auc(fpr, tpr)
    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]
    predicted_labels = np.zeros(len(real_labels))
    for i in range(len(predicted_probas)):
        if predicted_probas[i] > threshold:
            predicted_labels[i] = 1
    f1_final = f1_score(real_labels, predicted_labels)
    accuracy_final = accuracy_score(real_labels, predicted_labels)
    precision_final = precision_score(real_labels, predicted_labels)
    recall_final = recall_score(real_labels, predicted_labels)
    return [aupr_final, auc_final, f1_final, accuracy_final, precision_final, recall_final]

def GetCrossValidation(X, y, folds):
    predicted_probas = -np.ones(len(y))
    for train_index, test_index in folds:
        X_train, X_test, y_train = GetPartitionOfSamples(X, y, train_index, test_index)
        predict_test_proba = MakePrediction(X_train, X_test, y_train)
        predicted_probas[test_index] = predict_test_proba
    aupr_final, auc_final, f1_final, accuracy_final, precision_final, recall_final = EvaluatePerformances(y, predicted_probas)
    print(aupr_final, auc_final, f1_final, accuracy_final, precision_final, recall_final)
    return predicted_probas

def EnsembleLearning(X_train, X_test, y_train):
    input_num = len(X_train[0])
    dataset_size = len(X_train)
    batch_size = int(dataset_size * 1.0)
    hidden_num = 10

    x = tf.placeholder(tf.float32, [None, input_num])
    w1 = tf.Variable(tf.random_normal([input_num, hidden_num], stddev=1, seed=1))
    b1 = tf.Variable(tf.zeros([hidden_num]))
    w2 = tf.Variable(tf.random_normal([hidden_num, 1], stddev=1, seed=1))
    b2 = tf.Variable(tf.zeros([1]))
    y_ = tf.placeholder(tf.float32, [None, 1])
    my_hidden = tf.nn.relu(tf.matmul(x, w1) + b1)
    y = tf.nn.relu(tf.matmul(my_hidden, w2) + b2)

    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer(0.5).minimize(cross_entropy)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        steps = 300
        for i in range(steps):
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            sess.run(train_step, feed_dict={x: X_train[start:end], y_: y_train[start:end]})
        ensemble_test_proba = sess.run(y, feed_dict={x: X_test})
        return ensemble_test_proba

def Ensemble_CV(X, y, folds):
    ensemble_probas = -np.ones(len(y))
    for train_index, test_index in folds:
        X_train, X_test, y_train = GetPartitionOfSamples(X, y, train_index, test_index)
        ensemble_test_proba = EnsembleLearning(X_train, X_test, y_train)
        ensemble_probas[test_index] = ensemble_test_proba[:, 0]
    aupr_final, auc_final, f1_final, accuracy_final, precision_final, recall_final = EvaluatePerformances(y, ensemble_probas)
    return aupr_final, auc_final, f1_final, accuracy_final, precision_final, recall_final

########################################################################################
if __name__ == '__main__':
    all_features = ['SpectrumProfile', 'MismatchProfile', 'RevcKmer', 'PCP_SCP']
    posi_samples_file = 'SLT2_posi_samples.txt'
    nega_samples_file = 'SLT2_nega_samples.txt'
    fp = open(posi_samples_file, 'r')
    posis = GetSequences(fp)
    fn = open(nega_samples_file, 'r')
    negas = GetSequences(fn)
    y = array([[1]]*len(posis)+[[0]]*len(negas))

    folds_num = 5
    seeds_num = 100

    for seed in range(1, 1+seeds_num):
        folds = ConstructPartitionOfSet(y[:, 0], folds_num, seed)
        all_predicted_probas = []
        for feature in all_features:
            X = np.loadtxt(feature+'.txt')
            X = StandardScaler().fit(X, y).transform(X)
            clf = MLPClassifier(hidden_layer_sizes=(700),alpha = 0.3,random_state=1)
            predicted_probas = GetCrossValidation(X, y[:, 0], folds)
            all_predicted_probas.append(predicted_probas)

        all_predicted_probas = array(all_predicted_probas)
        all_predicted_probas = all_predicted_probas.T
        np.savetxt('all_predicted_probas_' + str(seed) + '.txt', all_predicted_probas)
        aupr_final, auc_final, f1_final, accuracy_final, precision_final, recall_final = Ensemble_CV(all_predicted_probas, y, folds)
        print(seed, aupr_final, auc_final, f1_final, accuracy_final, precision_final, recall_final)
        result = [aupr_final, auc_final, f1_final, accuracy_final, precision_final, recall_final]
        np.savetxt(str(seed) + '.txt', result)
