# -*- coding: utf-8 -*-
#   author: guifeng tang
import numpy as np
from numpy import array
from itertools import combinations_with_replacement, permutations
from repDNA.nac import RevcKmer
from repDNA.psenac import PCPseDNC, PCPseTNC, SCPseDNC, SCPseTNC
import time


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


def GetKmerDict(alphabet,k):
    kmerlst = []
    partkmers = list(combinations_with_replacement(alphabet, k))
    for element in partkmers:
        elelst = set(permutations(element, k))
        strlst = [''.join(ele) for ele in elelst]
        kmerlst += strlst
    kmerlst = np.sort(kmerlst)
    kmerdict = {kmerlst[i]:i for i in range(len(kmerlst))}
    return kmerdict


############################### Spectrum Profile ##############################
def GetSpectrumProfile(instances, alphabet, k):
    kmerdict = GetKmerDict(alphabet, k)
    X = []
    for sequence in instances:
        vector = GetSpectrumProfileVector(sequence, kmerdict, k)
        X.append(vector)
    X = array(X)
    return X


def GetSpectrumProfileVector(sequence, kmerdict, k):    
    vector = np.zeros((1, len(kmerdict)))
    n = len(sequence)
    for i in range(n-k+1):
        subsequence = sequence[i:i+k]
        position = kmerdict.get(subsequence)
        vector[0, position] += 1
    return list(vector[0])


############################### Mismatch Profile ##############################
def GetMismatchProfile(instances, alphabet, k, m):
    kmerdict = GetKmerDict(alphabet, k)
    X = []
    for sequence in instances:
        vector = GetMismatchProfileVector(sequence, alphabet, kmerdict, k)
        X.append(vector)  
    X = array(X)
    return X


def GetMismatchProfileVector(sequence, alphabet, kmerdict, k):    
    vector = np.zeros((1, len(kmerdict)))
    n = len(sequence)
    for i in range(n-k+1):
        subsequence = sequence[i:i+k]
        position = kmerdict.get(subsequence)
        vector[0, position] += 1
        for j in range(k):
            substitution = subsequence
            for letter in set(alphabet) ^ set(subsequence[j]):
                substitution = list(substitution)
                substitution[j] = letter
                substitution = ''.join(substitution)
                position = kmerdict.get(substitution)
                vector[0, position] += 1
    return list(vector[0])


########################### Reverse Compliment Kmer ###########################
def GetRevcKmer(k):
    rev_kmer = RevcKmer(k=k)
    pos_vec = rev_kmer.make_revckmer_vec(open(posi_samples_file))
    neg_vec = rev_kmer.make_revckmer_vec(open(nega_samples_file))
    X = array(pos_vec + neg_vec)
    return X


############ Parallel Correlation Pseudo Dinucleotide Composition #############
def GetPCPseDNC(lamada, phyche_list):
    pc_psednc = PCPseDNC(lamada=lamada, w=0.05)
    pos_vec = pc_psednc.make_pcpsednc_vec(open(posi_samples_file), phyche_index=phyche_list)
    neg_vec = pc_psednc.make_pcpsednc_vec(open(nega_samples_file), phyche_index=phyche_list)
    X = array(pos_vec + neg_vec)    
    return X


############ Parallel Correlation Pseudo Trinucleotide Composition ############
def GetPCPseTNC(lamada):
    pc_psetnc = PCPseTNC(lamada=lamada, w=0.05)
    pos_vec = pc_psetnc.make_pcpsetnc_vec(open(posi_samples_file), all_property=True)
    neg_vec = pc_psetnc.make_pcpsetnc_vec(open(nega_samples_file), all_property=True)
    X = array(pos_vec + neg_vec)
    return X


############## Series Correlation Pseudo Dinucleotide Composition #############
def GetSCPseDNC(lamada, phyche_list):
    sc_psednc = SCPseDNC(lamada=lamada, w=0.05)
    pos_vec = sc_psednc.make_scpsednc_vec(open(posi_samples_file), phyche_index=phyche_list)
    neg_vec = sc_psednc.make_scpsednc_vec(open(nega_samples_file), phyche_index=phyche_list)
    X = array(pos_vec + neg_vec)
    return X  


############## Series Correlation Pseudo Trinucleotide Composition ############
def GetSCPseTNC(lamada):
    sc_psetnc = SCPseTNC(lamada=lamada, w=0.05)
    pos_vec = sc_psetnc.make_scpsetnc_vec(open(posi_samples_file), all_property=True)
    neg_vec = sc_psetnc.make_scpsetnc_vec(open(nega_samples_file), all_property=True)
    X = array(pos_vec + neg_vec)
    return X 


###############################################################################
if __name__ == '__main__':
    global posi_samples_file
    global nega_samples_file
    posi_samples_file = 'SLT2_posi_samples.txt'
    nega_samples_file = 'SLT2_nega_samples.txt'
    fp = open(posi_samples_file, 'r')
    posis = GetSequences(fp)
    fn = open(nega_samples_file, 'r')
    negas = GetSequences(fn)
    instances = array(posis+negas)
    alphabet = ['A', 'C', 'G', 'T']
    
    # Spectrum Profile for k=1,2,3,4,5
    for k in range(1, 6):
        print('..........................................................................')
        print('Coding for feature:'+str(k)+'-Spectrum Profile, beginning')
        tic = time.clock()
        X = GetSpectrumProfile(instances, alphabet, k)
        np.savetxt(str(k)+'-SpectrumProfileFeature'+'.txt', X)
        toc = time.clock()
        print('Coding time:%.3f minutes' % ((toc-tic)/60))
        
    # Mismatch Profile for (k,m)=(3,1),(4,1),(5,1)
    for (k, m) in [(3, 1), (4, 1), (5, 1)]:
        print('..........................................................................')
        print('Coding for feature:'+str((k, m))+'-Mismatch Profile, beginning')
        tic = time.clock()
        X = GetMismatchProfile(instances, alphabet, k, m)
        np.savetxt(str((k, m))+'-MismatchProfileFeature'+'.txt', X)
        toc = time.clock()
        print('Coding time:%.3f minutes' % ((toc-tic)/60))

    # Reverse Compliment Kmer for k=1,2,3,4,5
    for k in range(1, 6):
        print('..........................................................................')
        print('Coding for feature:'+str(k)+'-RevcKmer, beginning')
        tic = time.clock()
        X = GetRevcKmer(k)
        np.savetxt(str(k)+'-RevcKmerFeature'+'.txt', X)
        toc = time.clock()
        print('Coding time:%.3f minutes' % ((toc-tic)/60))
        
    # Parallel Correlation Pseudo Dinucleotide Composition   
    print('..........................................................................')
    print('Coding for feature:PCPseDNC, beginning')
    tic = time.clock()
    X = GetPCPseDNC(9, phyche_list=['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'])
    np.savetxt('PCPseDNCFeature'+'.txt', X)
    toc = time.clock()
    print('Coding time:%.3f minutes' % ((toc-tic)/60))

    # Parallel Correlation Pseudo Trinucleotide Composition   
    print('..........................................................................')
    print('Coding for feature:PCPseTNC, beginning')
    tic = time.clock()
    X = GetPCPseTNC(1)
    np.savetxt('PCPseTNCFeature'+'.txt', X)
    toc = time.clock()
    print('Coding time:%.3f minutes' % ((toc-tic)/60))
    
    # Series Correlation Pseudo Dinucleotide Composition   
    print('..........................................................................')
    print('Coding for feature:SCPseDNC, beginning')
    tic = time.clock()
    X = GetSCPseDNC(15, phyche_list=['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise'])
    np.savetxt('SCPseDNCFeature'+'.txt', X)
    toc = time.clock()
    print('Coding time:%.3f minutes' % ((toc-tic)/60))
    
    # Series Correlation Pseudo Trinucleotide Composition   
    print('..........................................................................')
    print('Coding for feature:SCPseTNC, beginning')
    tic = time.clock()
    X = GetSCPseTNC(1)
    np.savetxt('SCPseTNCFeature'+'.txt', X)
    toc = time.clock()
    print('Coding time:%.3f minutes' % ((toc-tic)/60))

    # all features - 17
