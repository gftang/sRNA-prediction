# -*- coding: utf-8 -*-
#   author: guifeng tang
import numpy as np
from numpy import array
from pandas import DataFrame
import random
import time

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
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


def ConductCrossValidation(all_X, y, classifier_name, folds):
    predicted_proba = -np.ones(len(y))
    all_optimal_weights = []
    cv_round = 1
    for train_index, vali_index, test_index in folds:
        print('..........................................................................')
        print('Cross validation, round %d, beginning' % cv_round)
        start = time.clock()
        
        vali_proba_matrix, y_vali, test_proba_matrix = \
            GetIndividualFeatureResults(all_X, y, classifier_name, train_index, vali_index, test_index)
       
        optimal_weights = GeneticAlgorithm(vali_proba_matrix, y_vali)
        all_optimal_weights.append(optimal_weights)
        
        combined_proba = np.dot(test_proba_matrix, optimal_weights)
        predicted_proba[test_index] = combined_proba

        end = time.clock()
        print('The optimal weights:\n', optimal_weights)
        print('Round %d, running time: %.3f hour' % (cv_round, (end-start)/3600))
        print('..........................................................................\n')            
        cv_round += 1
    [aupr_score, auc_score, f1, accuracy, precision, recall] = EvaluatePerformances(y, predicted_proba)
    all_optimal_weights = array(all_optimal_weights)
    return aupr_score, auc_score, f1, accuracy, precision, recall, all_optimal_weights


def GetIndividualFeatureResults(all_X, y, classifier_name, train_index, vali_index, test_index):
    vali_proba_matrix = []
    test_proba_matrix = []
    for i in range(len(all_X)):
        X = all_X[i]
        if classifier_name == 'SVM':
            X = NormalizeFeature(X)
        clf = GetClassifier(classifier_name, all_features[i])
        X_train, X_vali, X_test, y_train, y_vali = \
            GetPartitionOfSamples(X, y, train_index, vali_index, test_index)

        classifier = clf.fit(X_train, y_train)
        temp_vali_proba = classifier.predict_proba(X_vali)
        temp_test_proba = classifier.predict_proba(X_test)
        
        vali_proba_matrix.append(temp_vali_proba[:, 1])
        test_proba_matrix.append(temp_test_proba[:, 1])
        
    vali_proba_matrix = np.transpose(vali_proba_matrix)
    test_proba_matrix = np.transpose(test_proba_matrix)
    return vali_proba_matrix, y_vali, test_proba_matrix


def GetPartitionOfSamples(X, y, train_index, vali_index, test_index):
    y_train = y[train_index]
    y_vali = y[vali_index]
    X_train = X[train_index]
    X_vali = X[vali_index]
    X_test = X[test_index]
    return X_train, X_vali, X_test, y_train, y_vali  


def GeneticAlgorithm(vali_proba_matrix, y_vali):
    global pops_num
    global generations
    global chr_length
    pops = GetPopulations(pops_num, chr_length)
    auc_scores = FitnessFunction(pops, vali_proba_matrix, y_vali)
    for k in range(generations):
        pops = Updatepops(pops,auc_scores)
        auc_scores = FitnessFunction(pops, vali_proba_matrix, y_vali)
    max_auc = np.max(auc_scores)
    print('The maximum AUC is %.3f on validation dataset' % max_auc)
    max_index = list(auc_scores).index(np.max(auc_scores))
    optimal_weights = pops[max_index]
    return optimal_weights


def GetPopulations(pops_num, chr_length):
    pops = []
    for i in range(pops_num-chr_length):
        temp_pop = [random.uniform(0, 1) for i in range(chr_length)]
        temp_pop = temp_pop/np.sum(temp_pop)
        pops.append(temp_pop)
    pops = array(pops)
    pops = np.vstack((np.eye(chr_length), pops))
    return pops


def FitnessFunction(pops, vali_proba_matrix, y_vali):
    auc_scores = []
    for i in range(np.shape(pops)[0]):
        weights = pops[i]
        combined_mean_proba = np.dot(vali_proba_matrix, weights)
        fpr, tpr, thresholds = roc_curve(y_vali, combined_mean_proba, pos_label=1)
        auc_scores.append(auc(fpr, tpr))          
    auc_scores = array(auc_scores)
    return auc_scores    


def Updatepops(pops, auc_scores):
    global pops_num
    new_order = random.sample(range(pops_num), pops_num)
    for i in np.linspace(0, pops_num, num=pops_num/2, endpoint=False, dtype=int):
        fmax = np.max(auc_scores)
        fmin = np.min(auc_scores)
        fmean = np.mean(auc_scores)
        
        select_index = new_order[i:i+2]
        f = np.max(auc_scores[select_index])
        two_pops = pops[select_index].copy()
        
        probacrossover = (fmax-f)/(fmax-fmean) if f > fmean else 1
        cross_pops = Crossover(two_pops) if probacrossover > random.uniform(0, 1) else two_pops.copy()

        probamutation = 0.5*(fmax-f)/(fmax-fmean) if f > fmean else (fmean-f)/(fmean-fmin)
        new_two_pops = Mutation(cross_pops) if probamutation > random.uniform(0, 1) else cross_pops.copy()
      
        pops[select_index] = new_two_pops.copy()
    return pops


def Crossover(two_pops):
    global chr_length
    cross_pops = two_pops.copy()
    crossposition = random.randint(2, chr_length-3)
    cross_pops[0][0:crossposition] = two_pops[1][0:crossposition]
    cross_pops[1][0:crossposition] = two_pops[0][0:crossposition]
    cross_pops = Normalize(cross_pops)
    return cross_pops


def Mutation(cross_pops):
    global chr_length
    new_two_pops = cross_pops.copy()
    for i in range(2):
        mutation_num = random.randint(1, round(chr_length/5))
        mutation_positions = random.sample(range(chr_length), mutation_num)
        new_two_pops[i][mutation_positions] = [random.uniform(0, 1) for j in range(mutation_num)]
    new_two_pops = Normalize(new_two_pops)
    return new_two_pops 


def Normalize(two_pops):
    global chr_length
    for i in range(2):     
        if np.sum(two_pops[i]) < 10**(-12):
            two_pops[i] = [random.uniform(0, 1) for j in range(chr_length)]
        two_pops[i] = two_pops[i]/np.sum(two_pops[i])
    return two_pops


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
    all_features = ['1-SpectrumProfile', '2-SpectrumProfile', '3-SpectrumProfile', \
                    '4-SpectrumProfile', '5-SpectrumProfile', '(3, 1)-MismatchProfile', \
                    '(4, 1)-MismatchProfile', '(5, 1)-MismatchProfile', '1-RevcKmer', \
                    '2-RevcKmer', '3-RevcKmer', '4-RevcKmer', '5-RevcKmer', \
                    'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']
    global chr_length
    global pops_num
    global generations
    chr_length = len(all_features)   #the length of chromosomes for the genetic algorithm
    pops_num = 100                  #the population size for the genetic algorithm
    generations = 200                #the generation for the genetic algorithm
    folds_num = 5                   #the number of folds for the cross validation
    seeds_num = 20                   #the number of seeds for the partition of dataset

    # classifier_name = 'RF'
    classifier_name = 'SVM'
    posi_samples_file = 'SLT2_posi_samples.txt '
    nega_samples_file = 'SLT2_nega_samples.txt'
    fp = open(posi_samples_file, 'r')
    posis = GetSequences(fp)
    fn = open(nega_samples_file, 'r')
    negas = GetSequences(fn)
    y = array([1]*len(posis)+[0]*len(negas))
    print('The number of positive and negative samples: %d,%d' % (len(posis), len(negas)))
    
    all_X = []
    for feature in all_features:
        X = np.loadtxt(feature+'Feature'+'.txt')
        all_X.append(X)
    
    results = []
    for seed in range(1, seeds_num+1):
        print('################################## Seed %d ##################################' % seed)
        print('The prediction using GA-based ensemble learning, beginning')
        print('This process may spend some time, please do not close the program')
        tic = time.clock()
              
        folds = ConstructPartitionOfSet(y, folds_num, seed)
        aupr_score, auc_score, f1, accuracy, precision, recall, all_optimal_weights = \
            ConductCrossValidation(all_X, y, classifier_name, folds)
        results.append([aupr_score, auc_score, f1, accuracy, precision, recall])
        
        toc = time.clock()
        print('**************************************************************************')
        print('Seed %d, the final predicted results:' % seed)
        print('*aupr score:%.3f, auc:%.3f, f1:%.3f, acc:%.3f, pre:%.3f, recall:%.3f*' \
              % (aupr_score, auc_score, f1, accuracy, precision, recall))
        print('Total running time:%.3f hour' % ((toc-tic)/3600))
        print('**************************************************************************\n')    
        
        feature_weights = DataFrame({'Round '+str(i+1): all_optimal_weights[i, :] for i in range(folds_num)})
        feature_weights['Feature'] = all_features
        feature_weights = feature_weights[['Feature']+['Round '+str(i+1) for i in range(folds_num)]]
        feature_weights.to_csv('OptimalFeatureWeights(seed%d).csv' % seed, index=False)
    
    results = array(results)
    df = DataFrame({'Seed': range(1, seeds_num+1), \
                    'aupr': results[:, 0], \
                    'auc': results[:, 1], \
                    'f1': results[:, 2], \
                    'acc': results[:, 3], \
                    'pre': results[:, 4], \
                    'recall': results[:, 5]})
    df = df[['Seed', 'aupr', 'auc', 'f1', 'acc', 'pre', 'recall']]
    df.to_csv('Results'+'CV(GAWE).csv', index=False)
