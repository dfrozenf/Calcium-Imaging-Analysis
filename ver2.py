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
filepath = 'C:/Users/User/Desktop/Calcium Image/Test/Results.csv'  # Specify the filepath of your imageJ output
pad = 30  # Needed to avoid falling off the function for signals near start and end of recording


def read(filepath, cell):
    with open(filepath, 'r') as f:
        input = f.readlines()
    input = [i.rstrip().split(sep=',') for i in input]
    frames = [int(i[0]) for i in input]
    bins = [float(i[1]) for i in input]
    signal = [float(i[cell + 1]) for i in input]

    return frames, bins, signal


def differ(signal, kernel, bins):
    signal = np.pad(signal, pad_width=pad, mode='edge')
    zsignal = medfilt(signal, kernel)
    dsignal = np.gradient(zsignal)
    ddsignal = np.gradient(dsignal)

    bins = np.pad(bins, pad_width=pad, mode='edge')
    dbins = np.gradient(bins)
    #    ddbins = np.gradient(dbins)

    return signal, zsignal, dsignal, ddsignal, dbins


def thresholder(q, dsignal):
    abs_dsignal = [abs(i) for i in dsignal]
    md = np.sum(abs_dsignal)/len(abs_dsignal)
    threshold = q * md

    return threshold


def aligner(thresh, dsignal, signal, ddsignal, dist=3):  # Set to 2 seconds
    pivots = find_peaks(dsignal, height=thresh, distance=dist)[0]

    align = np.zeros(len(pivots) * 60).reshape(len(pivots), 60)
    dalign = np.zeros(len(pivots) * 60).reshape(len(pivots), 60)
    ddalign = np.zeros(len(pivots) * 60).reshape(len(pivots), 60)

    for i in range(len(pivots)):
        align[i] = [signal[i] for i in range(int(pivots[i]) - 30, pivots[i] + 30)]
        dalign[i] = [dsignal[i] for i in range(int(pivots[i]) - 30, pivots[i] + 30)]
        ddalign[i] = [ddsignal[i] for i in range(int(pivots[i]) - 30, pivots[i] + 30)]

    return align, dalign, ddalign, pivots


def meanfunc(align, dalign, ddalign):
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

    return mean, dmean, ddmean,


def point_extract(dsignal):
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

    return [p1, p2, p3, p4]


def feature_extract(signal, dsignal, points):
    # TODO: Add roi size to PCA

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

for z in range(1, 614):

    frames, bins, signal = read(filepath, z)
    signal, zsignal, dsignal, ddsignal, dbins = differ(signal, 3, bins)
    timeon = find_peaks(dbins, distance=pad * 2)[0]
    nbins = len(timeon)
    thresh = thresholder(4.5, dsignal)

    #fig, (ax1, ax2) = plt.subplots(1,2)
    #ax1.hlines(thresh, xmin=0, xmax=len(dsignal))
    #ax1.plot([i for i in range(len(dsignal))], dsignal)
    #ax2.plot([i for i in range(len(zsignal))], zsignal)
    #plt.show()

    align, dalign, ddalign, pivots = aligner(thresh, dsignal, signal, ddsignal)
    if len(align) != 0:
        mean, dmean, ddmean = meanfunc(align, dalign, ddalign)

        point_names = ['P1', 'P2', 'P3', 'P4']

        #fig, (ax1, ax2) = plt.subplots(1, 2)

        for i in range(len(pivots)):
            points = point_extract(dalign[i])
            features = feature_extract(align[i], dalign[i], points)
            light = [j for j in timeon if j <= pivots[i]]

            if len(light) == 0:
                latency = -999
            elif light[-1] <= pad:
                latency = -999
            else:
                latency = pivots[i] - light[-1]

            features.append(latency)
            features.append(z)
            features.append(i)
            print(features)
            Features.append(features)
            print('Feature extraction completed on roi {}, signal {}'.format(z, i))
            #ax1.plot([i for i in range(len(dalign[i]))], dalign[i], alpha=0.5, color='grey')
            #ax2.plot([i for i in range(len(align[i]))], align[i], alpha=0.5, color='grey')
        #plt.show()
i = 0
while i < len(Features):
    if len(Features[i]) != 14:
        Features.pop(i)
    i += 1

np.savetxt('C:/Users/User/Desktop/calciumdump.csv', Features, delimiter=',', newline='\n')


def plot_align(dmean=dmean, dalign=dalign):
    for i in dalign:
        plt.plot([j for j in range(len(i))], i, alpha=0.5, color='grey')
    plt.plot([i for i in range(len(dmean))], dmean, alpha=0.8, color='black')
    d = interp1d([i for i in range(len(dmean))], dmean)
    for i in range(len(points)):
        plt.plot(points[i], d(points[i]), 'bo')
        plt.text(points[i], d(points[i]), point_names[i])
    plt.title('Aligned delF/F FD')
    plt.hlines(0, xmin=0, xmax=len(dmean))
    plt.show()


def plot_mean(signal=signal, dsignal=dsignal, timeon=timeon, mean=mean, dmean=dmean, ddmean=ddmean):
    plt.plot([i for i in range(len(signal))], signal)
    plt.vlines(timeon, ymin=min(signal), ymax=max(signal), linestyle='dashed', color='yellow')
    plt.show()

    plt.plot([i for i in range(len(dsignal))], dsignal)
    plt.hlines(thresh, xmin=0, xmax=len(dsignal))
    plt.title('delF/F FD signal, threshold={}'.format(thresh))
    plt.show()

    for i in align:
        plt.plot([j for j in range(len(i))], i, alpha=0.5, color='grey')
    plt.plot([i for i in range(len(mean))], mean, alpha=0.8, color='black')
    # plt.plot([i for i in range(len(zmean))], zmean, alpha=1, color='red')
    plt.title('Aligned delF/F')
    plt.show()

    for i in dalign:
        plt.plot([j for j in range(len(i))], i, alpha=0.5, color='grey')
    plt.plot([i for i in range(len(dmean))], dmean, alpha=0.8, color='black')
    d = interp1d([i for i in range(len(dmean))], dmean)
    for i in range(len(points)):
        plt.plot(points[i], d(points[i]), 'bo')
        plt.text(points[i], d(points[i]), point_names[i])
    plt.title('Aligned delF/F FD')
    plt.hlines(0, xmin=0, xmax=len(dmean))
    plt.show()

    plt.plot([i for i in range(len(ddmean))], ddmean)
    plt.title('Aligned delF/F SD')
    plt.show()

# plot_align()
