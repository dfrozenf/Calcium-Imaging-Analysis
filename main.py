import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt

filepath = 'C:/Users/User/Downloads/TestCalcium.csv'

def read(filepath, cell):
    with open(filepath, 'r') as f:
        input = f.readlines()
    input = [i.rstrip().split(sep=',') for i in input]
    frames = [int(i[0]) for i in input]
    bins = [float(i[1]) for i in input]
    signal = [float(i[cell + 1]) for i in input]

    return frames, bins, signal

def differ(signal, kernel):
    zsignal = medfilt(signal, kernel)
    dsignal = np.gradient(zsignal)
    ddsignal = np.gradient(dsignal)

    dbins = np.gradient(bins)
#    ddbins = np.gradient(dbins)

    return zsignal, dsignal, ddsignal, dbins

def align(thresh, dsignal, signal, ddsignal, dist=20):
    pivots = find_peaks(dsignal, height=thresh, distance=dist)[0]

    align = np.zeros(len(pivots)*40).reshape(len(pivots),40)
    dalign = np.zeros(len(pivots)*40).reshape(len(pivots),40)
    ddalign = np.zeros(len(pivots)*40).reshape(len(pivots),40)

    for i in range(len(pivots)):
        align[i] = [signal[i] for i in range(int(pivots[i])-20, pivots[i]+20)]
        dalign[i] = [dsignal[i] for i in range(int(pivots[i])-20, pivots[i]+20)]
        ddalign[i] = [ddsignal[i] for i in range(int(pivots[i])-20, pivots[i]+20)]

    return align, dalign, ddalign

def mean(align, dalign, ddalign, kernel = 3):
    mean = align[0]
    dmean = dalign[0]
    ddmean = ddalign[0]
    for i in range(1, len(align)):

        mean += (align[i])

        dmean += (dalign[i])

        ddmean += (ddalign[i])
    mean = mean/len(align)
    dmean = dmean/len(dalign)
    ddmean = ddmean/len(ddalign)

    return mean, dmean, ddmean,

def thresh(q, dsignal):
    sigman = np.median(abs(dsignal)/0.6745)
    thresh = q * sigman

    return thresh

def point_extract(zdmean, dmean):
    P1 = False
    i = 1
    while P1 == False:
        if dmean[i] == 0 or dmean[i-1] == 0:
            i+= 1
        elif dmean[i]/dmean[i-1] < 2:
            i+= 1
        else:
            P1 = i

    P2 = find_peaks(dmean, distance = len(dmean))[0][0]

    P3 = (np.diff(np.sign(zdmean[P2:])) != 0)*1
    P3 = np.where(P3)[0] + 21
    P3 = P3[0]

    P4 = find_peaks(-dmean, distance = len(dmean))[0][0]

    P5 = np.where((np.diff(np.sign(zdmean)) != 0)*1)[0][-1]

    return [P1, P2, P3, P4, P5]

frames, bins, signal = read(filepath,4)
dsignal, ddsignal, dbins, zsignal = differ(signal, kernel=3)
timeon = find_peaks(dbins, distance = 50)[0]
nbins = len(timeon)
thresh = thresh(1, dsignal)
align, dalign, ddalign = align(thresh, dsignal, signal, ddsignal, dist=20)
mean, dmean, ddmean = mean(align, dalign, ddalign, 7)
#points= point_extract(dmean, dmean)
#point_names = ['P1', 'P2', 'P3', 'P4', 'P5']

plt.plot([i for i in range(len(signal))], signal)
plt.vlines(timeon, ymin = min(signal), ymax = max(signal))
plt.show()

plt.plot([i for i in range(len(dsignal))], dsignal)
plt.hlines(thresh, xmin = 0, xmax = len(dsignal))
plt.title('delF/F FD signal, threshold={}'.format(thresh))
plt.show()

for i in align:
    plt.plot([j for j in range(len(i))], i, alpha=0.5, color='grey')
plt.plot([i for i in range(len(mean))], mean, alpha=0.8, color='black')
#plt.plot([i for i in range(len(zmean))], zmean, alpha=1, color='red')
plt.title('Aligned delF/F')
plt.show()

for i in dalign:
    plt.plot([j for j in range(len(i))], i, alpha=0.5, color='grey')
plt.plot([i for i in range(len(dmean))], dmean, alpha=0.8, color='black')
#for i in range(len(points)):
#    plt.plot(points[i], zdmean[points[i]], 'bo')
#    plt.text(points[i], zdmean[points[i]], point_names[i])
plt.title('Aligned delF/F FD')
plt.show()

plt.plot([i for i in range(len(ddmean))], ddmean)

plt.show()