import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

"""
Reads in imageJ outputs and dumps the extracted features into a csv for PCA
File input format: A .csv file where each column contains a separate roi's deltaF/F0 signal. 
                   First column is a numbered list used for frame count, no headers


"""
filepath = 'C:/Users/Daniel Frozenfar/Desktop/Calcium/rd1-Thy1-G6s/Results.csv'  # Specify the filepath of your imageJ output
roipath = 'C:/Users/Daniel Frozenfar/Desktop/Calcium/rd1-Thy1-G6s/ROI.csv'
pad = 30  # Needed to avoid falling off the function for signals near start and end of recording


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

with open(filepath, 'r') as f:
    f.readline()
    n_roi = f.readline()
    n_roi = [i for i in n_roi.rstrip().split(sep=',')]
n_roi = len(n_roi) - 1

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


def meanfunc(align, dalign, ddalign):
#Find the mean curve of the filtered, 1st derivative, and 2nd derivative calcium signals for plotting purposes
    mean = align[0]
    dmean = dalign[0]
    ddmean = ddalign[0]
    for i in range(1, len(align)):
        mean += (align[i])

        dmean += (dalign[i])

        ddmean += (ddalign[i])
    mean = mean / len(align)
    dmean = dmean / len(dalign)
    ddmean = ddmean / len(ddalign)
#mean, dmean, and ddmean are each a 1x60 array containing the mean value of the filtered, 1st derivative, and 2nd derivative deltaF/F0 respectively
    return mean, dmean, ddmean


def point_extract(dsignal):
#Given our chosen feature space, extract points P1, P2, P3, and P4 where P1 is the zero crossing before P2, 
#P2 is the peak of the signal, P3 is the zero crossing between P2 and P4, and P4 is the valley of the calcium signal
    zeros = np.where(np.diff(np.sign(dsignal)))[0]

    if max(zeros) <= 30:
        return [-9999]

    # p2 = find_peaks(dsignal, distance = len(dsignal))[0]
    p2 = 30

    i = p2 - 1
    while i not in zeros:
        i -= 1
    x = np.array([i, i + 1]).reshape((-1, 1))
    y = np.array([dsignal[i], dsignal[i + 1]])
    model = LinearRegression().fit(x, y)
    p1 = -model.intercept_ / model.coef_

    i = p2 + 1
    while i not in zeros:
        i += 1
    x = np.array([i, i + 1]).reshape((-1, 1))
    y = np.array([dsignal[i], dsignal[i + 1]])
    model = LinearRegression().fit(x, y)
    p3 = -model.intercept_ / model.coef_

    slice = -dsignal[p2:]

    try:
        p4 = find_peaks(slice, distance=len(slice))[0][0] + p2
    except IndexError:
        p4 = -9999
#Returns a length-4 list of the indices containing P1,P2,P3, and P4 in an aligned signal
    return [p1, p2, p3, p4]


def feature_extract(signal, dsignal, points):
#Extract features specified in features.docm from an aligned signal
    if -9999 in points:
        return []
    P1 = points[0]
    P2 = points[1]
    P3 = points[2]
    P4 = points[3]

    F1 = P4 - P2
    F2 = dsignal[P4] - dsignal[P2]
    F3 = F2 / F1

    d = interp1d([i for i in range(len(dsignal))], dsignal)
    F4 = ((dsignal[P2] - d(P1)) / (P2 - P1)) / ((d(P3) - dsignal[P2]) / (P3 - P2))[0]
    F5 = signal[P2]
    F6 = dsignal[P2]
    F7 = signal[P4]
    F8 = dsignal[P4]
    F9 = np.quantile(dsignal, .75) - np.quantile(dsignal, .25)

    F10 = 0
    dhat = np.mean(dsignal)
    for i in range(len(dsignal)):
        F10 += (dsignal[i] - dhat) ** 4
    F10 /= (len(dsignal) * (np.std(dsignal) ** 4))

    F11 = 0
    dhat = np.mean(dsignal)
    for i in range(len(dsignal)):
        F11 += (dsignal[i] - dhat) ** 3
    F11 /= (len(dsignal) * (np.std(dsignal) ** 3))

    return [F1, F2, F3, F4[0], F5, F6, F7, F8, F9, F10, F11]


Features = []

for z in range(1, n_roi): #Iterate through the number of roi's
#TODO: add a method for automatically adjusting this iterator

    fig, axs = plt.subplots(1, 3)

    with open(roipath) as f:
        f.readline()
        area = f.readline().split(sep=',')
    area = [int(i.rstrip()) for i in area]
    area = area[z]

    frames, bins, signal = read(filepath, z) #Read an roi's signal from the imageJ output
    signal, zsignal, dsignal, ddsignal, dbins = differ(signal, 3, bins) #Filter and differentiate the raw signal
    timeon = find_peaks(dbins, distance=pad * 2)[0] #Find the light timings for latency calculation
    nbins = len(timeon) #Determine the number of light stimulus in a recording
    thresh = thresholder(4.5, dsignal) #Calculate a threshold for 1-st derivative based alignment and feature extraction

    axs[0].plot([i for i in range(len(zsignal))], zsignal, linewidth=0.5)
    axs[0].set_xlim(30, 330)
    axs[0].set_ylim(0, 1.5)
    axs[0].set_ylabel('DeltaF/F0')
    axs[0].set_xlabel('Time (Frames)')
    axs[0].title.set_text('Raw Signal')

    axs[1].plot([i for i in range(len(dsignal))], dsignal, linewidth=0.5)
    axs[1].hlines(thresh, xmin = 0, xmax=len(dsignal), color='r', linestyles='dotted')
    axs[1].title.set_text('First Derivative')
    axs[1].set_xlim(30, 330)
    axs[1].set_ylim(-0.75, 1.2)
    axs[1].set_ylabel('d(DeltaF/F0)/dt')
    axs[1].set_xlabel('Time (Frames)')

    axs[2].plot(zsignal[30:331], dsignal[30:331], linewidth=0.5)
    axs[2].title.set_text('Phase Plot')
    axs[2].set_xlim(0, 1.5)
    axs[2].set_ylim(-1.2, 1.2)
    axs[2].set_xlabel('DeltaF/F0')
    axs[2].set_ylabel('d(DeltaF/F0)/dt')

    plt.show()

    #fig, (ax1, ax2) = plt.subplots(1,2)
    #ax1.hlines(thresh, xmin=0, xmax=len(dsignal))
    #ax1.plot([i for i in range(len(dsignal))], dsignal)
    #ax2.plot([i for i in range(len(zsignal))], zsignal)
    #plt.show()

    trim = (timeon[0]+timeon[1])//2
    signal = signal[trim:]
    dsignal = dsignal[trim:]
    ddsignal = ddsignal[trim:]

    align, dalign, ddalign, pivots = aligner(thresh, dsignal, signal, ddsignal) #Align signal and derivatives
    if len(align) != 0: #Error check for an roi containing no signals
        mean, dmean, ddmean = meanfunc(align, dalign, ddalign)

        point_names = ['P1', 'P2', 'P3', 'P4']

        #fig, (ax1, ax2) = plt.subplots(1, 2)

        for i in range(len(pivots)): #Extract all features for each signal in an alignment and append it to a global features matrix
            points = point_extract(dalign[i])
            features = feature_extract(align[i], dalign[i], points)
            light = [j for j in timeon if j <= pivots[i]]

            if len(light) == 0:
                latency = -999
            elif light[-1] <= pad:
                latency = -999
            else:
                latency = pivots[i] - light[-1]

            features.append(area)

            features.append(latency)
            features.append(z)
            features.append(i)
            features[-2] = features[-2] + features[-1]/10
            print(features)
            Features.append(features)
            print('Feature extraction completed on roi {}, signal {}'.format(z, i))
            #ax1.plot([i for i in range(len(dalign[i]))], dalign[i], alpha=0.5, color='grey')
            #ax2.plot([i for i in range(len(align[i]))], align[i], alpha=0.5, color='grey')
        #plt.show()
i = 0
while i < len(Features):
    if len(Features[i]) != 15:
        Features.pop(i)
    i += 1

np.savetxt('C:/Users/Daniel Frozenfar/Desktop/calciumdump.csv', Features, delimiter=',', newline='\n')