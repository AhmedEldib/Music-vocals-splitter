import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import wave
from songdata import song_data
import os
import sounddevice as sd
import scipy
from scipy.io.wavfile import write

from sklearn.decomposition import FastICA, PCA

from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget, plot, PlotItem
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from PyQt5 import QtCore
import pyaudio
from main_window import Ui_MainWindow

#python -m PyQt5.uic.pyuic -x main_window.ui -o main_window.py

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()

        self.songsArray = []
        pg.setConfigOption('background', 'k')

        self.PlayArray = []

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.out1 = []
        self.out2 = []

        self.ui.actionImport.triggered.connect(self.Import_song)
        self.ui.save_outputs.clicked.connect(self.Save)

        self.ui.Start_1.clicked.connect(self.Player_out1)
        self.ui.Start_2.clicked.connect(self.Player_out2)

        self.ui.Stop_1.clicked.connect(self.Player_stop)
        self.ui.Stop_2.clicked.connect(self.Player_stop)

    def Save(self):
        if len(self.out1) != 0:
            write("output1.wav", 44100, self.out1)
            write("output2.wav", 44100, self.out2)

    def Play_wav(self):
        if (len(self.PlayArray) != 0):
            sd.play(self.PlayArray, len(self.PlayArray)/self.songsArray[0].durationF)
        else:
            pass

    def Player_out1(self):
        self.PlayArray = np.array(self.out1)
        self.Play_wav()

    def Player_out2(self):
        self.PlayArray = np.array(self.out2)
        self.Play_wav()

    def Player_stop(self):
        sd.stop()

    def Import_song(self):
            self.songsArray = []
            filePaths = QtWidgets.QFileDialog.getOpenFileNames(self, 'Multiple File',"~/Desktop",'*.wav')
            for filePath in filePaths:
                for f in filePath:
                    print('filePath',f, '\n')
                    if f == '*' or f == None:
                        break

                    tempSong = song_data()
                    tempSong.set_Data(f)
                    self.songsArray.append(tempSong)
                    self.gen_output()

    def plot(self, graph_window, x_data, y_data):
        graph_window.clear()
        graph_window.plot(x_data, y_data, pen = 'w')

        Gmin = np.nanmin(y_data)
        Gmax = np.nanmax(y_data)
        graph_window.plotItem.getViewBox().setLimits(xMin=np.nanmin(x_data), xMax=np.nanmax(x_data), yMin=Gmin - Gmin * 0.1, yMax=Gmax + Gmax * 0.1)
        QtCore.QCoreApplication.processEvents()

    def gen_output(self):
        self.plot(self.ui.Inputview, self.songsArray[0].Input_fourier_Freq, self.songsArray[0].Input_fourier_Mag)

        self.plot(self.ui.Outview_1, self.songsArray[0].Input_fourier_Freq, self.songsArray[0].channel_1)

        self.plot(self.ui.Outview_2, self.songsArray[0].Input_fourier_Freq, self.songsArray[0].channel_2)

        s1 = self.songsArray[0].first_channel
        s2 = self.songsArray[0].second_channel


        S = np.c_[s1, s2]


        #A = np.array([[1, 1], [1, 1]])  # Mixing matrix
        X = S

        # Compute ICA
        ica = FastICA(n_components = 2)
        S_ = ica.fit_transform(X)  # Reconstruct signals
        A_ = ica.mixing_  # Get estimated mixing matrix

        # #############################################################################
        S_ = S_ * 1000000
        S_ = S_.astype(np.int16)

        self.out1 = S_[:, 0]
        self.out2 = S_[:, 1]


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
