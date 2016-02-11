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


################################################################################
# core program
#
# FFT time scales almost linearly with the largest factor...
#
FACTOR_LIMIT = 100
#
def bestFFTlength(n):
    while max(factorint(n)) >= FACTOR_LIMIT:
        n -= 1
    return n


# given an array of data and a sampling rate in Hz, return the
# frequency of ticks
#
def get_period(samples, rate, plot=False):
    n = samples.shape[0]
    minfreq = 1.0 * rate / n

    # plot the last 10 seconds of raw data
    #
    if plot:
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


    # look in the region between 0.5 and 10 Hz.  if your clock doesn't
    # tick in the frequency range, you may be in trouble
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
    if plot:
        plt.subplot(2,2,3)
        plt.plot(freqs,f1)
        plt.scatter(freqs[peaks],f1[peaks],s=50)

        plt.xlim([0,10])
        plt.ylim([0,1.1])

        plt.xlabel('frequency (Hz)')
        plt.ylabel('power')


    # get a crude guess to the frequency, and cut the data around it
    #
    fguess = np.min(peaks)

    lo = fguess-10
    hi = fguess+10

    f1    = f1[lo:hi]
    freqs = freqs[lo:hi]

    fguess = np.argsort(f1)[-1] # find_peaks() doesn't always get the
                                # exact location


    # update frequency guess based on gaussian interpolation
    # (ie, fit a parabola to log(f1))
    #
    alpha = log(f1[fguess-1])
    beta  = log(f1[fguess])
    gamma = log(f1[fguess+1])

    a = 0.5 * (alpha - 2*beta + gamma)
    p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
    b = beta - a * p**2


    # new plot, zoomed in on the peak, with a Gaussian fit to the profile
    #
    if plot:
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


    fguessrefined = freqs[fguess] + p * minfreq

    return freqs[fguess], fguessrefined


# take the filename of a sound recording of a watch.  report the tick
# frequency, along with the error estimate in helpful units such as
# seconds/day.  show a plot to assess quality of fit
#
def analyze_file(fname):
    # import the data into a numpy array
    #
    try:
        rate, samples = wavfile.read(fname)
        minfreq = 1.0 * rate / samples.shape[0]
    except:
        print "error reading file {0}".format(fname)
        exit(1)

    n = bestFFTlength(samples.shape[0])
    samples = samples[-n:]
    minfreq = 1.0 * rate / n

    fguess, fguessrefined = get_period(samples, rate, True)

    sigfigs = int(round(log10(fguess/minfreq))) + 1
    print "initial guess: {0} p/m {1:.3} Hz".format(round(fguess, sigfigs), minfreq)


    # estimate error and report results
    #
    # analytic estimate from CERN guys is garbage
    # sigma = 0.0516 * 0.01 * minfreq

    # repeat with half the data
    n = bestFFTlength(int(samples.shape[0]/2))
    samples = samples[-n:]
    minfreq = 1.0 * rate / n

    fguess2, fguessrefined2 = get_period(samples, rate)

    sigma = abs(fguessrefined2-fguessrefined) / 2**1.5
    sigfigs = int(round(log10(fguessrefined/sigma))) + 1

    print "updated guess: {0} p/m {1:.3} Hz".format(round(fguessrefined, sigfigs), sigma)


    # estimate the error and report
    #
    err = fguessrefined - round(fguessrefined)
    err = err/round(fguessrefined)

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



################################################################################
# code to generate synthetic tick signals for testing
#
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
    fest,ferr = get_period(data, rate)

    return freq, fest, (fest-freq), ferr


def geterr(dur, sigma):
    ntrials = 100
    results = np.zeros(ntrials)
    for i in range(ntrials):
        freq, fest, err, sigma = trial(44100, dur, 5.99, 6.01, sigma)
        results[i] = err/freq * (3600 * 24)

    return np.mean(results), np.std(results)

def test_accuracy():
    d = np.array([10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0])
    e = np.zeros(d.shape[0])
    s = np.zeros(d.shape[0])

    for i,dur in enumerate(d):
        print dur
        mean, err = geterr(dur, 0.6)
        e[i] = mean
        s[i] = err

    out = np.column_stack((d,e,s))

    np.savetxt('44.1k-6Hz-n0.3.dat',out)
    for i,dur in enumerate(d):
        print dur
        mean, err = geterr(dur, 0.6)
        e[i] = mean
        s[i] = err

    out = np.column_stack((d,e,s))
    np.savetxt('44.1k-6Hz-n0.3.dat',out)

    np.savetxt('44.1k-6Hz-n0.1.dat',out)
    for i,dur in enumerate(d):
        print dur
        mean, err = geterr(dur, 0.6)
        e[i] = mean
        s[i] = err

    out = np.column_stack((d,e,s))
    np.savetxt('44.1k-6Hz-n0.1.dat',out)

    np.savetxt('44.1k-6Hz-n0.03.dat',out)
    for i,dur in enumerate(d):
        print dur
        mean, err = geterr(dur, 0.6)
        e[i] = mean
        s[i] = err

    out = np.column_stack((d,e,s))
    np.savetxt('44.1k-6Hz-n0.03.dat',out)



################################################################################
# test accuracy using synthetic tick data generated with audacity
#
def analyze_audacity_file(fname):
    # import the data into a numpy array
    #
    try:
        rate, samples = wavfile.read(fname)
        minfreq = 1.0 * rate / samples.shape[0]
    except:
        print "error reading file {0}".format(fname)
        exit(1)

    # n = bestFFTlength(samples.shape[0])
    # samples = samples[-n:]
    # minfreq = 1.0 * rate / n

    fguess, fguessrefined = get_period(samples, rate, True)

    # estimate error and report results
    #
    # repeat with half the data
    n = bestFFTlength(int(samples.shape[0]/2))
    samples = samples[-n:]
    minfreq = 1.0 * rate / n

    fguess2, fguessrefined2 = get_period(samples, rate)

    sigma = abs(fguessrefined2-fguessrefined) / 2**1.5
    sigfigs = int(round(log10(fguessrefined/sigma))) + 1

    # estimate the error and report
    #
    err = (fguessrefined - 5.0)/5.0
    sigma = sigma/5.0

    return err, sigma

def report_audacity():
    files = ['5Hz-10sec.wav',
             '5Hz-15sec.wav',
             '5Hz-30sec.wav',
             '5Hz-1min.wav',
             '5Hz-2min.wav',
             '5Hz-5min.wav',
             '5Hz-10min.wav']

    durs = np.array([10, 15, 30, 60, 120, 300, 600]) * 1.0

    err   = np.zeros(durs.shape[0])
    sigma = np.zeros(durs.shape[0])

    for i, file in enumerate(files):
        e,s = analyze_audacity_file("test-signals/{0}".format(file))
        err[i] = e
        sigma[i] = s

    out = np.column_stack((durs, err, sigma))
    np.savetxt('audacity-5Hz-test.dat', out)



################################################################################
# "the program in itself"
#
fname = sys.argv[1]
analyze_file(fname)
plt.show()
