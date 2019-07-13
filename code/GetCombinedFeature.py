import numpy as np

all_features = ['1-SpectrumProfile', '2-SpectrumProfile', '3-SpectrumProfile', '4-SpectrumProfile', \
                '5-SpectrumProfile', '(3, 1)-MismatchProfile', '(4, 1)-MismatchProfile', \
                '(5, 1)-MismatchProfile', '1-RevcKmer', '2-RevcKmer', '3-RevcKmer', '4-RevcKmer', \
                '5-RevcKmer', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC']

SpectrumProfile = np.loadtxt(all_features[0] + 'Feature' + '.txt')
for i in range(1, 5):
    X = np.loadtxt(all_features[i] + 'Feature' + '.txt')
    SpectrumProfile = np.hstack((SpectrumProfile, X))
np.savetxt('SpectrumProfile' + '.txt', SpectrumProfile)
print SpectrumProfile.shape

MismatchProfile = np.loadtxt(all_features[5] + 'Feature' + '.txt')
for i in range(6, 8):
    X = np.loadtxt(all_features[i] + 'Feature' + '.txt')
    MismatchProfile = np.hstack((MismatchProfile, X))
np.savetxt('MismatchProfile' + '.txt', MismatchProfile)
print MismatchProfile.shape

RevcKmer = np.loadtxt(all_features[8] + 'Feature' + '.txt')
for i in range(9, 13):
    X = np.loadtxt(all_features[i] + 'Feature' + '.txt')
    RevcKmer = np.hstack((RevcKmer, X))
np.savetxt('RevcKmer' + '.txt', RevcKmer)
print RevcKmer.shape

PCP_SCP = np.loadtxt(all_features[13] + 'Feature' + '.txt')
for i in range(14, 17):
    X = np.loadtxt(all_features[i] + 'Feature' + '.txt')
    PCP_SCP = np.hstack((PCP_SCP, X))
np.savetxt('PCP_SCP' + '.txt', PCP_SCP)
print PCP_SCP.shape
