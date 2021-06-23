import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from sklearn.cluster import AgglomerativeClustering
from rdp import rdp

filepath = 'C:/Users/Daniel Frozenfar/Desktop/PCtransform.csv'
pad = 30

input = pd.read_csv(filepath)
clusters = AgglomerativeClustering().fit_predict(input)

filepath = 'C:/Users/Daniel Frozenfar/Desktop/Calcium/rd1-Thy1-G6s/Results.csv'  # Specify the filepath of your imageJ output

def read(filepath, cell):
#Given the filepath of the imageJ output and a selected roi number (can be iterated through), return a list of deltaF/F0 values observed in the roi
    with open(filepath, 'r') as f:
        f.readline()
        input = f.readlines()
    input = [i.rstrip().split(sep=',') for i in input]
    frames = [int(i[0]) for i in input]
    bins = [float(i[1]) for i in input]
    signal = [float(i[cell + 1]) for i in input]
#frames is a numbered list that we can use as a timeseries while plotting calcium signals
#bins is the 0 cell, by convention the 0-th roi will contain background signal, used to calculate light stimulus timing
#signal is the actual observed deltaF/F0 observed in a given recording
    return frames, bins, signal


def differ(signal, kernel, bins):
#Given a raw signal, the light stimulus, and a kernel for filtering, return a median filtered raw signal and the first & second derivatives of the filtered signal
    signal = np.pad(signal, pad_width=pad, mode='edge')
    zsignal = medfilt(signal, kernel)
    dsignal = np.gradient(zsignal)
    ddsignal = np.gradient(dsignal)

    bins = np.pad(bins, pad_width=pad, mode='edge')
    dbins = np.gradient(bins)
    #    ddbins = np.gradient(dbins)
#zsignal refers to the median filtered raw signal
#dsignal refers to the first derivative of the zsignal
#ddsignal refers to the second derivative of the zsignal
#dbins refers to the derivative of the light stimulus signal
    return signal, zsignal, dsignal, ddsignal, dbins


def thresholder(q, dsignal):
#Given some multiplier q (optimal value in citation was 4.5), calculate a threshold based off of mean deviation
    abs_dsignal = [abs(i) for i in dsignal]
    md = np.sum(abs_dsignal)/len(abs_dsignal)
    threshold = q * md
#threshold provides the nevessary deltaF/F0 threshold for our peakfinding function
    return threshold


def aligner(thresh, dsignal, signal, ddsignal, dist=3):  # Set to 2 seconds
#Given the calculated threshold, find peaks around a 2 second window and return alignment of the zsignal, d
    pivots = find_peaks(dsignal, height=thresh, distance=dist)[0]

    align = np.zeros(len(pivots) * 60).reshape(len(pivots), 60)
    dalign = np.zeros(len(pivots) * 60).reshape(len(pivots), 60)
    ddalign = np.zeros(len(pivots) * 60).reshape(len(pivots), 60)

    for i in range(len(pivots)):
        align[i] = [zsignal[i] for i in range(int(pivots[i]) - 30, pivots[i] + 30)]
        dalign[i] = [dsignal[i] for i in range(int(pivots[i]) - 30, pivots[i] + 30)]
        ddalign[i] = [ddsignal[i] for i in range(int(pivots[i]) - 30, pivots[i] + 30)]
#align, dalign, and ddalign are n x 60 matrices, each row holding the filtered, 1st derivative, and 2nd derivative values of an individual spike
#pivots gives us a global reference that we need to refer back from alignments when calcualting latency
    return align, dalign, ddalign, pivots

cluster_0 = []
dcluster_0 = []
cluster_1 = []
dcluster_1 = []

for z in range(len(clusters)):

    signal_id = input['Unnamed: 0'][z]
    signal_id = [math.floor(signal_id), int(math.ceil((signal_id - math.floor(signal_id)) * 10))]

    cluster_id = clusters[z]

    frames, bins, signal = read(filepath, signal_id[0])
    signal, zsignal, dsignal, ddsignal, dbins = differ(signal, 3, bins)
    thresh = thresholder(4.5, dsignal)

    align, dalign, ddalign, pivots = aligner(thresh, dsignal, signal, ddsignal)# Align signal and derivatives

    cmap = ['r','b']
    plt.plot(align[signal_id[1]-1], dalign[signal_id[1]-1],
             c='grey', alpha=1, linewidth=0.05)
    if cluster_id == 0:
        cluster_0.append(align[signal_id[1]-1])
        dcluster_0.append(dalign[signal_id[1]-1])
    elif cluster_id == 1:
        cluster_1.append(align[signal_id[1] - 1])
        dcluster_1.append(dalign[signal_id[1] - 1])

cluster_0 = np.array(cluster_0).reshape((1,-1))
dcluster_0 = np.array(dcluster_0).reshape((1,-1))
cluster_1 = np.array(cluster_1).reshape((1,-1))
dcluster_1 = np.array(dcluster_1).reshape((1,-1))

plt.title('All phase plots')
plt.ylabel('First Derivative of DeltaF/F0 w.r.t. Time')
plt.xlabel('DeltaF/F0')
plt.xscale('symlog')
plt.show()

eps= 0.25

cl_0 = np.array(rdp([[cluster_0[0][i], dcluster_0[0][i]] for i in range(len(cluster_0[0]))], epsilon=eps)).transpose()
cl_1 = np.array(rdp([[cluster_1[0][i], dcluster_1[0][i]] for i in range(len(cluster_1[0]))], epsilon=eps)).transpose()

plt.title('Clustering Results')
plt.ylabel('First Derivative of DeltaF/F0 w.r.t. Time')
plt.xlabel('DeltaF/F0')
plt.xscale('symlog')
plt.plot(cl_0[0], cl_0[1], c='lightcoral', linewidth=0.5, alpha=0.8, label='Cluster 1')
plt.plot(cl_1[0], cl_1[1], c='skyblue', linewidth=0.5, alpha=0.8, label='Cluster 2')
plt.legend()
plt.show()


#plt.plot(cluster_0, dcluster_0, c='lightcoral')
#plt.plot(cluster_1, dcluster_1, c='skyblue')
#plt.title('Average phase of each cluster')
#plt.ylabel('First Derivative of DeltaF/F0 w.r.t. Time')
#plt.xlabel('DeltaF/F0')
#plt.show()