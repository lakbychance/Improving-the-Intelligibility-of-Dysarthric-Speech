from __future__ import division
from scipy.io import wavfile
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind
import pprint as pp
import operator



# Input Signals (f_wave1=Control    f_wave2=Dysarthric)
f_wave1= "E:\Arpan Roy\Amity\Final Project 2017-18\Improving the intelligibility of dysarthric speech\Audio\Control\CF02\CF02_B1_C1_M3.wav"
f_wave2= "E:\Arpan Roy\Amity\Final Project 2017-18\Improving the intelligibility of dysarthric speech\Audio\F02\F02_B1_C1_M3.wav"

# read signal_1 (Control)
fs1, signal_1 = wavfile.read(f_wave1)
signal_1 = signal_1 / max(abs(signal_1))
assert min(signal_1) >= -1 and max(signal_1) <= 1

# read signal_2 (Dysarthric)
fs2, signal_2 = wavfile.read(f_wave2)
signal_2 = signal_2 / max(abs(signal_2))
assert min(signal_2) >= -1 and max(signal_2) <= 1

# Display the Sampling Frequency and the No. of Samples
print 'fs1     ==>', fs1, 'Hz'
print 'Length1 ==>', len(signal_1), 'samples'
print ''
print 'fs2     ==>', fs2, 'Hz'
print 'Length2 ==>', len(signal_2), 'samples'

# Plot Raw Signal

# Control
plt.subplot(3,2,1)
plt.plot(signal_1)    # for 'x' samples plt.plot(signal[0:x])
plt.title('Control (wrd="Command")')
plt.xlabel('(a) SAMPLES')
plt.autoscale(tight='both')

# Dysarthric
plt.subplot(3,2,2)
plt.plot(signal_2)    # for 'x' samples plt.plot(signal[0:x])
plt.title('Dysarthric (wrd="Command")')
plt.xlabel('(a) SAMPLES')
plt.autoscale(tight='both')

# Feature Extraction

# Control
[Fs1, x1] = audioBasicIO.readAudioFile(f_wave1)
F1 = audioFeatureExtraction.stFeatureExtraction(x1, Fs1, 0.050*Fs1, 0.025*Fs1)
plt.subplot(3,2,3)
plt.plot(F1[0,:])
plt.title('ZCR')
plt.xlabel('Frames')
plt.ylabel('(b) ZCR')
plt.subplot(3,2,5)
plt.plot(F1[9,:])
plt.title('MFCC')
plt.xlabel('(c) Frames')
plt.ylabel('MFCC')

# Dysarthric
[Fs2, x2] = audioBasicIO.readAudioFile(f_wave2)
F2 = audioFeatureExtraction.stFeatureExtraction(x2, Fs2, 0.050*Fs2, 0.025*Fs2)
plt.subplot(3,2,4)
plt.plot(F2[0,100:200])
plt.title('ZCR')
plt.xlabel('Frames')
plt.ylabel('(b) ZCR')
plt.subplot(3,2,6)
plt.plot(F2[9,:])
plt.title('MFCC')
plt.xlabel('(c) Frames')
plt.ylabel('MFCC')

#Discriminating the features
def disc_feature(F1,F2):
    temp ={}
    for i in range(0,34):
        Energy_control = F1[i,:]
        Energy_dys = F2[i,:]
        data = {'C':Energy_control,'D':Energy_dys}
        for list1, list2 in combinations(data.keys(), 2):
            t, p = ttest_ind(data[list1], data[list2])
            temp[i]=p
    return temp
disc = disc_feature(F1,F2)
sorted_disc = sorted(disc.items(), key=operator.itemgetter(1))
pp.pprint(sorted_disc)

#Save the Extracted Features in a file
np.savetxt("C:\Users\Arpan Roy\Desktop\_Feature_file_Control.csv", F1, delimiter=",")
np.savetxt("C:\Users\Arpan Roy\Desktop\_Feature_file_Dysarthria.csv", F2, delimiter=",")

# Display the Plots
plt.show()










