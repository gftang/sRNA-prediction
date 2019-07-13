# -*- coding: utf-8 -*-
#   author: guifeng tang
import numpy as np
from numpy import array
from pandas import DataFrame
import time

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


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
    folds_temp = list(KFold(len(y), n_folds=folds_num, shuffle=True, random_state=np.random.RandomState(seed)))
    folds = []
    for i in range(folds_num):
        test_index = folds_temp[i][1]
        vali_index = folds_temp[(i+1) % folds_num][1]
        train_index = array(list(set(folds_temp[i][0]) ^ set(vali_index)))
        folds.append((train_index, vali_index, test_index))
    return folds


def GetCrossValidation(X, y, folds):
    predicted_probas = -np.ones(len(y))
    cv_round = 1
    for train_index, vali_index, test_index in folds:
        X_train, X_vali, X_test, y_train, y_vali = \
            GetPartitionOfSamples(X, y, train_index, vali_index, test_index)
        predict_test_proba = MakePrediction(X_train, X_vali, X_test, y_train, y_vali, cv_round)
        predicted_probas[test_index] = predict_test_proba
        cv_round += 1
    aupr_score, auc_score, f1, accuracy, precision, recall = EvaluatePerformances(y, predicted_probas)
    return aupr_score, auc_score, f1, accuracy, precision, recall


def GetPartitionOfSamples(X, y, train_index, vali_index, test_index):
    y_train = y[train_index]
    y_vali = y[vali_index]
    X_train = X[train_index]
    X_vali = X[vali_index]
    X_test = X[test_index]
    return X_train, X_vali, X_test, y_train, y_vali


def MakePrediction(X_train, X_vali, X_test, y_train, y_vali, cv_round):
    classifier = clf.fit(X_train, y_train)
    predict_vali_proba = classifier.predict_proba(X_vali)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_vali, predict_vali_proba, pos_label=1)
    auc_score = auc(fpr, tpr)
    print('Cross validation,round %d,the AUC is %.3f on validation dataset' % (cv_round, auc_score))
    predict_test_proba = classifier.predict_proba(X_test)[:, 1]
    return predict_test_proba


def EvaluatePerformances(real_labels, predicted_probas):
    precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probas, pos_label=1)
    aupr_score = auc(recall, precision)
    fpr, tpr, auc_thresholds = roc_curve(real_labels, predicted_probas)
    auc_score = auc(fpr, tpr)
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
    f1 = f1_score(real_labels, predicted_labels)
    accuracy = accuracy_score(real_labels, predicted_labels)
    precision = precision_score(real_labels, predicted_labels)
    recall = recall_score(real_labels, predicted_labels)
    print aupr_score, auc_score, f1, accuracy, precision, recall
    return [aupr_score, auc_score, f1, accuracy, precision, recall]


def GetClassifier(classifier_name, feature):
    all_gamma = {'1-SpectrumProfile': 100, '2-SpectrumProfile': 10,\
                 '3-SpectrumProfile': 1, '4-SpectrumProfile': 0.1,\
                 '5-SpectrumProfile': 0.01, '(3, 1)-MismatchProfile': 10,\
                 '(4, 1)-MismatchProfile': 1, '(5, 1)-MismatchProfile': 0.1,\
                 '1-RevcKmer': 1000, '2-RevcKmer': 100, '3-RevcKmer': 10,
                 '4-RevcKmer': 1, '5-RevcKmer': 0.1, 'PCPseDNC': 1,
                 'PCPseTNC': 1, 'SCPseDNC': 1, 'SCPseTNC': 1}

    if classifier_name == 'RF':
        clf = RandomForestClassifier(random_state=1, n_estimators=200)
    elif classifier_name == 'SVM':
        clf = svm.SVC(kernel='rbf', gamma=all_gamma[feature], probability=True)
    elif classifier_name == 'LR':
        clf = LogisticRegression()
    return clf


def NormalizeFeature(X):
    X = X + 0.0
    for i in range(X.shape[1]):
        max_value = max(X[:, i])
        min_value = min(X[:, i])
        X[:, i] = (X[:, i] - min_value) / (max_value - min_value)
    return X


########################################################################################
if __name__ == '__main__':
    all_features = ['1-SpectrumProfile', '2-SpectrumProfile', '3-SpectrumProfile', '4-SpectrumProfile', \
                    '5-SpectrumProfile', '(3, 1)-MismatchProfile', '(4, 1)-MismatchProfile', \
                    '(5, 1)-MismatchProfile', '1-RevcKmer', '2-RevcKmer', '3-RevcKmer', '4-RevcKmer', \
                    '5-RevcKmer', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']
    classifier_name = 'RF'
    # classifier_name = 'SVM'
    posis = GetSequences(open('SLT2_posi_samples.txt', 'r'))
    negas = GetSequences(open('SLT2_nega_samples.txt', 'r'))
    y = array([1]*len(posis)+[0]*len(negas))

    print('The number of positive and negative samples: %d,%d' % (len(posis), len(negas)))

    folds_num = 5
    seeds_num = 20

    average_results = 0
    for seed in range(1, 1+seeds_num):
        print('################################# Seed %d ###################################' % seed)
        start = time.clock()
        folds = ConstructPartitionOfSet(y, folds_num, seed)
        results = []
        for feature in all_features:
            print('.............................................................................')          
            print('The prediction based on feature:'+feature+', beginning')
            tic = time.clock()
            X = np.loadtxt(feature+'Feature'+'.txt')
            if classifier_name == 'SVM':
                X = NormalizeFeature(X)
            clf = GetClassifier(classifier_name, feature)
            print('The dimension of the '+feature+':%d' % len(X[0]))
            aupr_score, auc_score, f1, accuracy, precision, recall = GetCrossValidation(X, y, folds)
            results.append([aupr_score, auc_score, f1, accuracy, precision, recall])
            toc = time.clock()
            print('*****************************************************************************')  
            print('The final results for feature:'+feature)
            print('*aupr:%.3f, auc:%.3f, f1:%.3f, acc:%.3f, pre:%.3f, recall:%.3f*' \
                  % (aupr_score, auc_score, f1, accuracy, precision, recall))
            print('Running time:%.3f mimutes' % ((toc-tic)/60))
            print('*****************************************************************************')      
            print('.............................................................................\n')    

        results = array(results)
        df = DataFrame({'Feature': all_features, \
                        'aupr': results[:, 0], \
                        'auc': results[:, 1], \
                        'f1': results[:, 2], \
                        'acc': results[:, 3], \
                        'pre': results[:, 4], \
                        'recall': results[:, 5]})
        df = df[['Feature', 'aupr', 'auc', 'f1', 'acc', 'pre', 'recall']]
        df.to_csv('IndividualFeatureResults'+'CV(seed'+str(seed)+')'+classifier_name+'.csv', index=False)
    
        end = time.clock()
        print('Seed %d, total running time:%.3f minutes' % (seed, (end-start)/60))
        print('#############################################################################')
        average_results += results

    average_results = average_results / seeds_num
    average_df = DataFrame({'Feature': all_features, \
                            'aupr': average_results[:, 0], \
                            'auc': average_results[:, 1], \
                            'f1': average_results[:, 2], \
                            'acc': average_results[:, 3], \
                            'pre': average_results[:, 4], \
                            'recall': average_results[:, 5]})
    average_df = average_df[['Feature', 'aupr', 'auc', 'f1', 'acc', 'pre', 'recall']]
    average_df.to_csv('IndividualFeatureAverageResults'+'CV('+classifier_name+').csv', index=False)
