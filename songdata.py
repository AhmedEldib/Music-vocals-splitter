import numpy as np
import pyqtgraph as pg
import logging
import sys
import wave
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import pylab
from skimage.feature import peak_local_max
import imagehash as ih
import scipy
from scipy import fftpack
from PIL import Image
import seaborn as sns
from scipy.io.wavfile import write

from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget, plot, PlotItem
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from PyQt5 import QtCore
import pyaudio
import scipy.io.wavfile as wav

class song_data():
    def __init__(self):
        self.song_path = None

        self.song_data = None
        self.frameRate = None
        

    def set_Data(self, filepath):
        if filepath != None:
            self.song_path = filepath
            self.get_song()
            self.calc_fourier()
            self.gen_channels()

    def get_wav_info(self,wav_file,channel=0):
        (freq, sig) = wav.read(wav_file)
        if sig.ndim == 2:
            return (sig[:,channel], freq)
        return (sig, freq)

    def get_song(self):
        head , tail = os.path.split(self.song_path)
        self.songName = tail
        self.songName = os.path.splitext(self.songName)[0]

        song = wave.open(self.song_path, 'r')
        #self.frameRate = song.getframerate()
        self.frame = song.getnframes()
        #self.frameRate = self.frameRate / 2
        song.close()
        
        sound_info, frame_rate = self.get_wav_info(self.song_path)
        frame_rate = 44100
        self.time = np.arange(0, self.frame) * (1.0 / frame_rate)
        

        
        self.song_data = sound_info
        self.frameRate = frame_rate
        self.durationF = self.frame / float(self.frameRate)

    #------------------------------------------------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------------------------------------------------

    def gen_channels(self):

        self.channel_1 = []
        self.channel_2 = []

        for i in range(len(self.Input_fourier_Mag)):
            
            if self.Input_fourier_Freq[i] < 1000 and self.Input_fourier_Freq[i] > -1000:
                self.channel_1.append(self.Input_fourier_Mag[i] * 0)
                self.channel_2.append(self.Input_fourier_Mag[i] * 0.01)
            
            elif self.Input_fourier_Freq[i] < 5000 and self.Input_fourier_Freq[i] > -5000:
                self.channel_1.append(self.Input_fourier_Mag[i] * 0.01)
                self.channel_2.append(self.Input_fourier_Mag[i] * 0)

            else:
                self.channel_1.append(0)
                self.channel_2.append(0)

        self.first_channel = self.Calc_inv_fourier(self.channel_1)
        self.second_channel = self.Calc_inv_fourier(self.channel_2)


    def calc_fourier(self):
        print(self.song_data)
        self.Input_fourier = scipy.fft(self.song_data)
        self.fourier_phase = np.angle(self.Input_fourier)
        self.Input_fourier_Mag = abs(scipy.fft(self.song_data))
        self.Input_fourier_Freq = scipy.fftpack.fftfreq(self.song_data.size, self.time[1] - self.time[0])

    def Calc_inv_fourier(self, new_mag):

        New_signal = np.multiply(new_mag, np.exp(1j*self.fourier_phase))
        inv_fourier_signal = np.real(np.fft.ifft(New_signal))

        inv_fourier_signal = inv_fourier_signal.astype(np.int16)  
        return inv_fourier_signal 

    def plot(self, graph_window, x_data, y_data):
        graph_window.clear()
        graph_window.plot(x_data, y_data, pen = 'w')

        Gmin = np.nanmin(y_data)
        Gmax = np.nanmax(y_data)
        graph_window.plotItem.getViewBox().setLimits(xMin=np.nanmin(x_data), xMax=np.nanmax(x_data), yMin=Gmin - Gmin * 0.1, yMax=Gmax + Gmax * 0.1)
        QtCore.QCoreApplication.processEvents()