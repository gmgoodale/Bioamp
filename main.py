# NI imports
import nidaqmx
from nidaqmx.stream_readers import (AnalogSingleChannelReader, AnalogMultiChannelReader)
from nidaqmx.constants import (AcquisitionType, CountDirection, Edge,
    READ_ALL_AVAILABLE, TaskMode, TriggerType, TerminalConfiguration)

# Kivy imports
from kivy.app import App
from kivy.lang import Builder
from kivy.garden.graph import MeshLinePlot, Graph
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.properties import (NumericProperty, ReferenceListProperty,
    ObjectProperty)
from kivy.uix.boxlayout import BoxLayout

# General imports
import collections
import numpy
import time
from threading import Thread
import threading
from copy import copy
from datetime import datetime
import os
from scipy.fftpack import fft

class FFT:

    # Takes in a deq in the time domain and returns a list of tuples (freq, mag)
    @staticmethod
    def FFTFromDEQ(timeDEQs, numChannels, samplingRate):
        # Convert to a numpy array and take the fft for each channel
        chFrequencies = []
        for DEQ in timeDEQs:
            timeArray = numpy.asarray(DEQ)[:,1]
            chFrequencies.append(numpy.abs(fft(timeArray)))

        # Generate the X axis values which are the discrete frequency values
        N = len(chFrequencies[0])
        n = numpy.arange(N)
        T = N/samplingRate
        freqValues = n/T

        freqGraphs = []
        # Generate a list of tuples from the X and Y values for graphing
        for ch in chFrequencies:
            freqGraph = []
            for x, y in zip(freqValues, ch):
                freqGraph.append((x,y))

            freqGraphs.append(freqGraph)

        return freqGraphs

class FileHandling:

    def __init__(self, patientNumber, numChannels):
        self.patientNumber = str(patientNumber)
        self.numChannels = numChannels
        self.fileName = str(patientNumber) + "_" + str(datetime.now().strftime("%Y_%m_%d %I_%M")) + ".csv"
        self.directory = str(os.getcwd()) + "\\Data\\" + str(patientNumber) + "\\"

        # Make the directory for the patient if it doesn't exist already
        if (not os.path.isdir(self.directory)):
            os.mkdir(self.directory)

        # Create the file, I suppose this could be done inline instead
        self.CreateFile()

    def CreateFile(self):
        self.file = open(self.toRaw(self.directory + self.fileName), "x")

        headerLine = "Time"
        for channel in range(self.numChannels):
            headerLine = ''.join([headerLine, ",Ch", str(channel)])

        headerLine = ''.join([headerLine, "\n"])
        self.file.write(headerLine)

    def SaveData(self, time, samplesRead):
        dataLine = str(time)

        for i in range(self.numChannels):
            dataLine = ''.join([dataLine, ",", str(samplesRead[i][0])])

        dataLine = ''.join([dataLine, "\n"])
        self.file.write(dataLine)


    def Close(self):
        # Flush the buffer and ensure everything is saved to disk before closing
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()

    def toRaw(self, string):
        return fr"{string}"

# Responsible for getting data from the NI DAQ and storing it
# If a new DAQ is used, write a new DAQ class and as long as it has channel
# buffers as deques it will be compatible with all of the code
class NIDAQ:

    def __init__(self, device, numChannels = 1, samplingRate = 5, histLen = 5):
        self.device = device
        self.samplingRate = samplingRate  # Sampling rate in Hz
        self.historyLength =  histLen # Number of samples in buffer to be displayed
        self.numberOfChannles = numChannels # Assumes channels are 0 -> numberOfChannles
        self.minVal = -1 # Sets the range for the DAQ in volts, reducing range increases precision
        self.maxVal = 1
        self.timeElapsed = 0 # Keeps track of how long data has been recorded

        # This creates a list of FIFO buffers of a fixed size, these buffers are how you access the channel data
        self.channelBuffers = []
        for i in range(self.numberOfChannles):
            # deque(dataype, maxlen of deque), note [(0,0)] is a list of tuples
            self.channelBuffers.append(collections.deque([(0, 0)], self.historyLength))

        # TESTING THIS WILL BE MOVED LATER
        #self.file = FileHandling("01", self.numberOfChannles)

    # Must be called called on an individual thread to handle constant updating
    def updateChannelsContinuosly(self):
        # Create a new task to perform the reading, this task will die when this method ends
        with nidaqmx.Task() as readTask:
            # Add all of the channels up to the self.number of channels
            for i in range(self.numberOfChannles):
                # RSE = reference single ended
                readTask.ai_channels.add_ai_voltage_chan(self.device + "ai" + str(i),
                    max_val=self.maxVal, min_val=self.minVal,
                    terminal_config=TerminalConfiguration.RSE)

            # This ensures that the DAQ is constantly sampling without prompt
            readTask.timing.cfg_samp_clk_timing(self.samplingRate,
                sample_mode=AcquisitionType.CONTINUOUS)
            readTask.start()

            # Stream reading allows for more elegant acquition at high rates
            self.reader = AnalogMultiChannelReader(readTask.in_stream)
            self.reader.verify_array_shape = False

            # Streamreader requires a numpy array to save values to
            holder_array = numpy.zeros((self.numberOfChannles, 1),
                dtype=numpy.float64)

            # This allows the thread to be stopped from the function that called it
            NIDAQThread = threading.currentThread()
            while getattr(NIDAQThread, "continueRunning", True):
                # Returns the number of samples read (same for each channel)
                # Waits until 1 sample is availabe and then reads it. Althrough
                # it seems nicer to read all available samples, nidaq currently has buffer sizing issues
                samplesRead = self.reader.read_many_sample(holder_array, number_of_samples_per_channel=1)
                self.timeElapsed += (1/self.samplingRate)

                #self.file.SaveData(self.timeElapsed, holder_array)

                for i in range(self.numberOfChannles):
                        self.channelBuffers[i].append((self.timeElapsed, holder_array[i][0]))
                        #print("Channel " + str(i) + ": " + str(self.channelBuffers[i]), flush=True)

            #self.file.Close()

# Responsible for displaying and updating the data to graph
class GraphValues(Widget):
    time_graph = ObjectProperty(None)
    frequency_graph = ObjectProperty(None)
    xMin = NumericProperty(0)
    xMax = NumericProperty(1)

    # Init must take in *kwargs for some reason. Something to do with inheriting from Widget
    def __init__(self, NIDevice, **kwargs):
        super(GraphValues, self).__init__(**kwargs)
        self.channelsToPlot = [0, 1]
        self.DAQSampleRate = 500
        self.histLen = 5000

        # This instantiation is important, it sets up the number of channels, buffer size and things of the like
        self.DAQ = NIDAQ(NIDevice, numChannels = len(self.channelsToPlot),
                samplingRate = self.DAQSampleRate, histLen = self.histLen)

        # Add the plots to the time_graph
        self.timePlots = []
        for ch in self.channelsToPlot:
            plot = MeshLinePlot(color=self.ColorGenerator(ch))
            self.timePlots.append(plot)
            self.time_graph.add_plot(plot)

        # Add a single plot to the frequency_graph
        self.freqPlots = []
        for ch in self.channelsToPlot:
            plot = MeshLinePlot(color=self.ColorGenerator(ch))
            self.freqPlots.append(plot)
            self.frequency_graph.add_plot(plot)


    # Starts the DAQ and the plotting, reserves the DAQ and the buffers
    def start(self):
        # This reseres the DAQ and it continusly gathers voltages in the buffers
        self.DAQThread = Thread(target=self.DAQ.updateChannelsContinuosly)
        self.DAQThread.start()

        # This gets the graph to update every 0.02 seconds
        Clock.schedule_interval(self.updateGraph, 0.02)

        # Disable the start button to avoid trying to reserve the resource again
        self.ids.start_button.disabled = True
        self.ids.stop_button.disabled = False

    def stop(self):
        # This stops the loop updating the buffer and waits for the thread to finish
        self.DAQThread.continueRunning = False
        self.DAQThread.join()

        # This stops the graph from updating
        Clock.unschedule(self.updateGraph)

        # Re-enable the start Button, disable stop button for aesthetics
        self.ids.start_button.disabled = False
        self.ids.stop_button.disabled = True

    # dt is update time interval and must be passed to any funciton called from clock
    def updateGraph(self, dt):
        print("updating...", flush=True)
        # Buffer copy to avoid read conflict with buffer writing
        tempBuffers = copy(self.DAQ.channelBuffers)
        ffts = FFT.FFTFromDEQ(tempBuffers, len(self.channelsToPlot), self.DAQSampleRate)

        # Update the x boundaries to "follow" the graph
        self.xMin = tempBuffers[0][0][0]
        self.xMax = tempBuffers[0][0][0] + (self.histLen/self.DAQSampleRate)

        for timePlot, timeVals, freqPlot, freqVals in zip(self.timePlots,
                tempBuffers, self.freqPlots, ffts):
            timePlot.points = timeVals
            freqPlot.points = freqVals

    # This just returns a unique color for the first 5 channels, if there is a better way to do this please do
    def ColorGenerator(self, ch):
        if (ch == 0):
            return [0, 1, 1, 1]
        elif (ch == 1):
            return [1, 0, 0, 1]
        elif (ch == 2):
            return [0, 1, 0, 1]
        elif (ch == 3):
            return [0, 0, 1, 1]
        elif (ch == 4):
            return [1, 0, 1, 1]
        else:
            return [1, 1, 1, 1]

# This is the main GUI function and must be named after the kivy file. (e.g. DaqApp -> Daq.kv)
class DaqApp(App, BoxLayout):
    def build(self):
        return GraphValues("Dev1/")

if __name__ == '__main__':
    DaqApp().run()
