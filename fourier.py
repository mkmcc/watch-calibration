#/usr/bin/env python
#
import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal
# wtf... numpy and scipy ffts not the same?
#from scipy.fftpack import rfft,rfftfreq
from numpy.fft import rfft,rfftfreq
from math import log,log10

import matplotlib.pyplot as plt

from sympy.ntheory import factorint


# FFT time scales almost linearly with the largest factor...
#
FACTOR_LIMIT = 100
#
def bestFFTlength(n):
    while max(factorint(n)) >= FACTOR_LIMIT:
        n -= 1
    return n


# import the data into a numpy array
#
fname = sys.argv[1]
try:
    rate, samples = wavfile.read(fname)
    minfreq = 1.0 * rate / samples.shape[0]
except:
    print "error reading file {0}".format(fname)
    exit(1)


n = bestFFTlength(samples.shape[0])
samples = samples[-n:]
minfreq = 1.0 * rate / n


# plot the raw data
#
plt.figure(0)
plt.subplot(2,1,1)
plt.plot(np.linspace(0,10,int(10*rate)), samples[-int(10*rate):])

plt.xlabel('time (s)')
plt.ylabel('amplitude')


# apply a threshold to cut out the noise
#
thresh = 3.0 * np.std(samples)
samples = np.fabs(samples)

idx = samples < thresh
samples[idx] = 0.0


# fourier transform the data to identify frequencies
# ... window it with a gaussian envelope to make gaussian-shaped peaks
# ... these can be accurately fit with gaussian profiles to centroid
# ... precise frequency
#
w = signal.gaussian(samples.shape[0], std=samples.shape[0]/7)

f1 = rfft(samples*w)
f1 = np.abs(f1)
f1 = f1 / np.max(f1)

freqs = rfftfreq(samples.shape[0], 1.0/rate)


# look in the region between 0.5 and 10 Hz
#
lo = np.argmax(freqs >  0.5)    # or just calculate from minfreq...
hi = np.argmax(freqs > 10.0)

f1    = f1[lo:hi]
freqs = freqs[lo:hi]


# arbitrary threshold here...
#
tmp = np.select([f1>0.5*np.max(f1)], [f1])
peaks = signal.find_peaks_cwt(tmp, np.arange(3,5))


# plot the fourier transform, along with identified peaks
#
plt.subplot(2,2,3)
plt.plot(freqs,f1)
plt.scatter(freqs[peaks],f1[peaks],s=50)

plt.xlim([0,10])
plt.ylim([0,1.1])

plt.xlabel('frequency (Hz)')
plt.ylabel('power')


# and plot again refined on the maximum
#
fguess=np.min(peaks)

lo=fguess-10
hi=fguess+10

f1 = f1[lo:hi]
freqs = freqs[lo:hi]

fguess=np.argsort(f1)[-1]

sigfigs = int(round(log10(freqs[fguess]/minfreq))) + 1
print "initial guess: {0} p/m {1:.3} Hz".format(round(freqs[fguess], sigfigs), minfreq)

alpha = log(f1[fguess-1])
beta  = log(f1[fguess])
gamma = log(f1[fguess+1])

a = 0.5 * (alpha - 2*beta + gamma)
p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
b = beta - a * p**2

xs = np.arange(0,f1.shape[0],0.1)
model = np.exp(a*(xs-p-fguess)**2 + b)

plt.subplot(2,2,4)
plt.plot(freqs,f1)
plt.plot(xs*minfreq+freqs[0],model)
plt.scatter(freqs[fguess],f1[fguess])
plt.scatter(minfreq*(p+fguess) + freqs[0],np.max(model), color='r')

plt.ylim([0,1.1])
plt.xlim([np.min(freqs),np.max(freqs)])

plt.xlabel('frequency (Hz)')
plt.ylabel('power')


# update frequency guess based on gaussian interpolation
#
corr = log(f1[fguess+1]/f1[fguess-1])
corr /= 2*log(f1[fguess]**2 / (f1[fguess+1]*f1[fguess-1]))

fguess = freqs[fguess] + corr * minfreq


# estimate error and report results
#
sigma = 0.0516 * 0.01 * minfreq
sigfigs = int(round(log10(fguess/sigma))) + 1

print "updated guess: {0} p/m {1:.3} Hz".format(round(fguess, sigfigs), sigma)


# estimate the error and report
#
err = fguess - round(fguess)    # hope the nearest integer is the
                                # intended tick rate.  if not, god it's bad...
err = err/round(fguess)

if err**2 <= sigma**2:
    err = sigma
    print "error consistent with zero.  below is an upper limit."
    print "record for a longer time and repeat"
    print ""

if abs(err) * 60 >= 1:
    print "error is {0:.1f} seconds / minute".format(err*60)
elif abs(err) * 3600 >= 1:
    print "error is {0:.1f} seconds / hour".format(err*3600)
elif abs(err) * 3600*24 >= 1:
    print "error is {0:.1f} seconds / day".format(err*3600*24)
elif abs(err) * 3600*24*30 >= 1:
    print "error is {0:.1f} seconds / month".format(err*3600*24*30)
elif abs(err) * 3600*24*365.25 >= 1:
    print "error is {0:.1f} seconds / year".format(err*3600*24*365.25)
else:
    print "it's perfect"

plt.show()
