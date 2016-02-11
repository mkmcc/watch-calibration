#/usr/bin/env python
#
import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal
# wtf... numpy and scipy ffts not the same?
#from scipy.fftpack import rfft,rfftfreq
from numpy.fft import rfft,rfftfreq
from math import log,log10,pi

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


def measure_tick_freq(samples, rate):

    # first, trim to data to a size acceptable for FFTs
    n = bestFFTlength(samples.shape[0])
    samples = samples[-n:]
    minfreq = 1.0 * rate / n


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


    # and plot again refined on the maximum
    #
    fguess=np.min(peaks)

    lo=fguess-10
    hi=fguess+10

    f1 = f1[lo:hi]
    freqs = freqs[lo:hi]

    fguess=np.argsort(f1)[-1]

    # update frequency guess based on gaussian interpolation
    #
    corr = log(f1[fguess+1]/f1[fguess-1])
    corr /= 2*log(f1[fguess]**2 / (f1[fguess+1]*f1[fguess-1]))

    fguess = freqs[fguess] + corr * minfreq


    # estimate error and report results
    #
    sigma = 0.0516 * 0.01 * minfreq * 20
    sigfigs = int(round(log10(fguess/sigma))) + 1

    return fguess, sigma


# make a single tick sound
#
def mk_tick(rate, duration, freq):
    n = duration * rate
    t = np.arange(n) / rate     # t is in sec

    phase = np.random.uniform(-pi, pi)

    y = np.cos(2*pi*freq * t - phase) * np.exp(-t/duration)

    return y


# make a train of ticks
#
def make_synthetic_data(rate, duration, freq, noise):
    tick_freq = 2.0e3           # Hz
    tick_dur  = 0.01            # s

    n = int(rate * duration)

    data = np.random.normal(0.0, noise, n)
    time = 1.0*np.arange(n)/rate    # units of seconds

    nticks = int(duration * freq)-1
    tick_times = (np.arange(nticks) + np.random.uniform(0,1))/freq

    for t in tick_times:
        i = int(round(t*rate))

        r1   = np.random.uniform(0.8, 1.2)
        r2   = np.random.uniform(0.8, 1.2)
        tick = mk_tick(rate, r1*tick_dur, r2*tick_freq)

        data[i:i+tick.shape[0]] += tick

    return time,data


def trial(rate, dur, fmin, fmax, noise):
    freq  = np.random.uniform(fmin, fmax)

    time,data = make_synthetic_data(rate, dur, freq, noise)
    fest,ferr = measure_tick_freq(data, rate)

    return freq, fest, (fest-freq), ferr


def geterr(dur):
    ntrials = 100
    results = np.zeros(ntrials)
    for i in range(ntrials):
        freq, fest, err, sigma = trial(44100, dur, 5.99, 6.01, 0.6)
        results[i] = err/freq * (3600 * 24)

    return np.mean(results), np.std(results)


d = np.array([10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0])
e = np.zeros(d.shape[0])
s = np.zeros(d.shape[0])
for i,dur in enumerate(d):
    print dur
    mean, err = geterr(dur)
    e[i] = mean
    s[i] = err

plt.figure(0)
plt.errorbar(d, e, s)

plt.figure(1)
plt.loglog(d, s)
plt.show()

out = np.column_stack((d,e,s))
np.savetxt('44.1k-6Hz-n0.6.dat',out)

# def geterr(rate):
#     ntrials = 100
#     results = np.zeros(ntrials)
#     for i in range(ntrials):
#         freq, fest, err, sigma = trial(rate, 60.0, 5.99, 6.01, 0.1)
#         results[i] = err/freq * (3600 * 24)

#     return np.mean(results), np.std(results)


# r = np.array([44100, 32000, 16000, 14500, 8000, 3600])
# e = np.zeros(r.shape[0])
# s = np.zeros(r.shape[0])
# for i,rate in enumerate(r):
#     print rate
#     mean, err = geterr(rate)
#     e[i] = mean
#     s[i] = err

# plt.figure(0)
# plt.errorbar(r, e, s)

# plt.figure(1)
# plt.loglog(r, s)
# plt.show()

# out = np.column_stack((r,e,s))
# np.savetxt('60s-6Hz-n0.1.dat',out)
