# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from scipy import signal


No = 100
dtt = 0.01
Tot = No * dtt
t = np.arange(0, Tot, dtt)
Amp = 1
phase = 0
frequency = 10

fig = plt.figure(figsize = (12, 10))
sig = fig.add_axes([0.07, 0.6, 0.25, 0.25])
fft = fig.add_axes([0.4, 0.2, 0.25, 0.25])
ifft = fig.add_axes([0.73, 0.2, 0.25, 0.25])
power = fig.add_axes([0.08, 0.2, 0.25, 0.25])

sig.set_ylim(-3, 3)
ifft.set_ylim(-3, 3)

y = Amp * np.sin(2*np.pi*frequency*t) + phase
sig.plot(t, y, linestyle = '-', marker = '.', color = 'blue')

F = np.fft.fft(y)
fft.plot(t, F.real, label = 'Real')
fft.plot(t, F.imag, label = 'Imaginary')
fft.legend()

iF = np.fft.ifft(F)
ifft.plot(t, iF, linestyle = '-', marker = '.', color = 'red')

FP1 = F[:(No//2)]
k1 = np.arange(0, No//2)/Tot
P1 = np.abs((FP1.real*FP1.imag)/No**2)
power.plot(k1, P1, linestyle = '-', marker = '.', color = 'green')

sig.set_title('Signal')
sig.set_xlabel('Time [s]')
sig.set_ylabel('Amplitude')
fft.set_title('Fourier transform')
fft.set_xlabel('Frequency [Hz]')
fft.set_ylabel('Amplitude')
ifft.set_title('Inverse fourier transform')
ifft.set_xlabel('Time [s]')
ifft.set_ylabel('Amplitude')
power.set_title('Power spectrum')
power.set_xlabel('Frequency [Hz]')
power.set_ylabel('$Amplitude^2$')

z = 0

def sinewave(freq, N, T, A, phi):
    sig.cla()
    dt = T/N
    t = np.arange(0, sliderHandleT.val, dt)
    y = A * np.sin(2*np.pi*freq*t - phi)
    sig.plot(t, y, linestyle = '-', marker = '.', color = 'blue')
    sig.set_title('Signal')
    sig.set_xlabel('Time [s]')
    sig.set_ylabel('Amplitude')
    sig.set_ylim(-3, 3)
    
def sawtooth(freq, N, T, A, phi):
    sig.cla()
    dt = T/N
    t = np.arange(0, sliderHandleT.val, dt)
    y = A * signal.sawtooth(2*np.pi*freq*t - phi)
    sig.plot(t, y, linestyle = '-', marker = '.', color = 'blue')
    sig.set_title('Signal')
    sig.set_xlabel('Time [s]')
    sig.set_ylabel('Amplitude')
    sig.set_ylim(-3, 3)
    
def square(freq, N, T, A, phi):
    sig.cla()
    dt = T/N
    t = np.arange(0, sliderHandleT.val, dt)
    y = A * signal.square(2*np.pi*freq*t - phi)
    sig.plot(t, y, linestyle = '-', marker = '.', color = 'blue')
    sig.set_title('Signal')
    sig.set_xlabel('Time [s]')
    sig.set_ylabel('Amplitude')
    sig.set_ylim(-3, 3)
    
def fouriersine(freq, N, T, A, phi):
    fft.cla()
    dt = T/N
    t = np.arange(0, T, dt)
    y = A * np.sin(2*np.pi*freq*t - phi)
    F = np.fft.fft(y)
    fft.plot(t, F.real, label = 'Real')
    fft.plot(t, F.imag, label = 'Imaginary')
    fft.legend()
    fft.set_title('Fourier transform')
    fft.set_xlabel('Frequency [Hz]')
    fft.set_ylabel('Amplitude')
    
def fouriersawtooth(freq, N, T, A, phi):
    fft.cla()
    dt = T/N
    t = np.arange(0, T, dt)
    y = A * signal.sawtooth(2*np.pi*freq*t - phi)
    F = np.fft.fft(y)
    fft.plot(t, F.real, label = 'Real')
    fft.plot(t, F.imag, label = 'Imaginary')
    fft.legend()
    fft.set_title('Fourier transform')
    fft.set_xlabel('Frequency [Hz]')
    fft.set_ylabel('Amplitude')
    
def fouriersquare(freq, N, T, A, phi):
    fft.cla()
    dt = T/N
    t = np.arange(0, T, dt)
    y = A * signal.square(2*np.pi*freq*t - phi)
    F = np.fft.fft(y)
    fft.plot(t, F.real, label = 'Real')
    fft.plot(t, F.imag, label = 'Imaginary')
    fft.legend()
    fft.set_title('Fourier transform')
    fft.set_xlabel('Frequency [Hz]')
    fft.set_ylabel('Amplitude')
    
def infouriersine(freq, N, T, A, phi):
    ifft.cla()
    dt = T/N
    t = np.arange(0, T, dt)
    y = A * np.sin(2*np.pi*freq*t - phi)
    sig.plot(t, y)
    F = np.fft.fft(y)
    iF = np.fft.ifft(F)
    ifft.plot(t, iF, linestyle = '-', marker = '.', color = 'red')
    ifft.set_title('Inverse fourier transform')
    ifft.set_xlabel('Time [s]')
    ifft.set_ylabel('Amplitude')
    ifft.set_ylim(-3, 3)
    
def infouriersawtooth(freq, N, T, A, phi):
    ifft.cla()
    dt = T/N
    t = np.arange(0, T, dt)
    y = A * signal.sawtooth(2*np.pi*freq*t - phi)
    sig.plot(t, y)
    F = np.fft.fft(y)
    iF = np.fft.ifft(F)
    ifft.plot(t, iF, linestyle = '-', marker = '.', color = 'red')
    ifft.set_title('Inverse fourier transform')
    ifft.set_xlabel('Time [s]')
    ifft.set_ylabel('Amplitude')
    ifft.set_ylim(-3, 3)
    
def infouriersquare(freq, N, T, A, phi):
    ifft.cla()
    dt = T/N
    t = np.arange(0, T, dt)
    y = A * signal.square(2*np.pi*freq*t - phi)
    sig.plot(t, y)
    F = np.fft.fft(y)
    iF = np.fft.ifft(F)
    ifft.plot(t, iF, linestyle = '-', marker = '.', color = 'red')
    ifft.set_title('Inverse fourier transform')
    ifft.set_xlabel('Time [s]')
    ifft.set_ylabel('Amplitude')
    ifft.set_ylim(-3, 3)
    
def powerspecsine(freq, N, T, A, phi):
    power.cla()
    dt = T/N
    t = np.arange(0, T, dt)
    y = A * np.sin(2*np.pi*freq*t - phi)
    F = np.fft.fft(y)
    FP = F[:(N//2)]
    k = np.arange(0, N//2)/T
    P = np.abs((FP.real*FP.imag)/N**2)
    power.plot(k, P, linestyle = '-', marker = '.', color = 'green')
    power.set_title('Power spectrum')
    power.set_xlabel('Frequency [Hz]')
    power.set_ylabel('$Amplitude^2$')

def powerspecsawtooth(freq, N, T, A, phi):
    power.cla()
    dt = T/N
    t = np.arange(0, T, dt)
    y = A * signal.sawtooth(2*np.pi*freq*t - phi)
    F = np.fft.fft(y)
    FP = F[:(N//2)]
    k = np.arange(0, N//2)/T
    P = np.abs((FP.real*FP.imag)/N**2)
    power.plot(k, P, linestyle = '-', marker = '.', color = 'green')
    power.set_title('Power spectrum')
    power.set_xlabel('Frequency [Hz]')
    power.set_ylabel('$Amplitude^2$')
    
def powerspecsquare(freq, N, T, A, phi):
    power.cla()
    dt = T/N
    t = np.arange(0, T, dt)
    y = A * signal.square(2*np.pi*freq*t - phi)
    F = np.fft.fft(y)
    FP = F[:(N//2)]
    k = np.arange(0, N//2)/T
    P = np.abs((FP.real*FP.imag)/N**2)
    power.plot(k, P, linestyle = '-', marker = '.', color = 'green')
    power.set_title('Power spectrum')
    power.set_xlabel('Frequency [Hz]')
    power.set_ylabel('$Amplitude^2$') 

def sine1(u):
    sinewave(sliderHandlefreq.val, sliderHandleN.val, sliderHandleT.val, sliderHandleA.val, sliderHandlephi.val)
    fouriersine(sliderHandlefreq.val, sliderHandleN.val, sliderHandleT.val, sliderHandleA.val, sliderHandlephi.val)
    infouriersine(sliderHandlefreq.val, sliderHandleN.val, sliderHandleT.val, sliderHandleA.val, sliderHandlephi.val)
    powerspecsine(sliderHandlefreq.val, sliderHandleN.val, sliderHandleT.val, sliderHandleA.val, sliderHandlephi.val)
    
def sawtooth1(u):
    sawtooth(sliderHandlefreq.val, sliderHandleN.val, sliderHandleT.val, sliderHandleA.val, sliderHandlephi.val)
    fouriersawtooth(sliderHandlefreq.val, sliderHandleN.val, sliderHandleT.val, sliderHandleA.val, sliderHandlephi.val)
    infouriersawtooth(sliderHandlefreq.val, sliderHandleN.val, sliderHandleT.val, sliderHandleA.val, sliderHandlephi.val)
    powerspecsawtooth(sliderHandlefreq.val, sliderHandleN.val, sliderHandleT.val, sliderHandleA.val, sliderHandlephi.val)   

def square1(u):
    square(sliderHandlefreq.val, sliderHandleN.val, sliderHandleT.val, sliderHandleA.val, sliderHandlephi.val)
    fouriersquare(sliderHandlefreq.val, sliderHandleN.val, sliderHandleT.val, sliderHandleA.val, sliderHandlephi.val)
    infouriersquare(sliderHandlefreq.val, sliderHandleN.val, sliderHandleT.val, sliderHandleA.val, sliderHandlephi.val)
    powerspecsquare(sliderHandlefreq.val, sliderHandleN.val, sliderHandleT.val, sliderHandleA.val, sliderHandlephi.val)

def wavetype(x):
    if x == 'Sine':
          sine1(0)
          z = 0
    elif x == 'Sawtooth':
          sawtooth1(0)
          z = 1
    elif x == 'Square':
        square1(0)
        z = 2
    return z

def slide(x):
    z = wavetype(radioHandle.value_selected)
    if z == 0:
        sine1(0)
    elif z == 1:
        sawtooth1(0)
    elif z == 2:
        square1(0)


rax = plt.axes([0.7, 0.87, 0.1, 0.1])
radioHandle = widgets.RadioButtons(rax, ['Sine', 'Sawtooth', 'Square'])
radioHandle.on_clicked(wavetype)

sax = plt.axes([0.5, 0.6, 0.4, 0.03])
sliderHandlefreq = widgets.Slider(sax, 'Frequency [Hz]', 0, 40, valinit=10)
sliderHandlefreq.on_changed(slide)

sax2 = plt.axes([0.5, 0.65, 0.4, 0.03])
sliderHandleN = widgets.Slider(sax2, 'Sampling points', 100, 500, valinit=100, valstep = 1)
sliderHandleN.on_changed(slide)

sax3 = plt.axes([0.5, 0.7, 0.4, 0.03])
sliderHandleT = widgets.Slider(sax3, 'Time [s]', 0, 2, valinit=1)
sliderHandleT.on_changed(slide)

sax4 = plt.axes([0.5, 0.75, 0.4, 0.03])
sliderHandleA = widgets.Slider(sax4, 'Amplitude', 0, 3, valinit=1)
sliderHandleA.on_changed(slide)

sax5 = plt.axes([0.5, 0.8, 0.4, 0.03])
sliderHandlephi = widgets.Slider(sax5, 'Phase[rad]', 0, 2*np.pi, valinit=0)
sliderHandlephi.on_changed(slide)

rax.text(-0.8, 1.1, 'Double click to change wave type')

def closeCallback(event):
    plt.close('all')
    
bax = plt.axes([0.87, 0.87, 0.1, 0.1]) 
buttonHandle = widgets.Button(bax, 'Close')
buttonHandle.on_clicked(closeCallback)