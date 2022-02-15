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
import multiprocessing
from copy import copy
from datetime import datetime
import os
from scipy.fftpack import fft
import queue

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

        for sample in samplesRead:
            dataLine = ''.join([dataLine, ",", str(sample)])

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
        self.minVal = -2 # Sets the range for the DAQ in volts, reducing range increases precision
        self.maxVal = 2
        self.timeElapsed = 0 # Keeps track of how long data has been recorded
        self.sampsAtATime = 4 # This sets the number of samples to grab at a time
        self.FFTLen = 2*histLen

        # This creates a list of FIFO buffers of a fixed size, these buffers are how you access the channel data
        self.channelBuffers = []
        for i in range(self.numberOfChannles):
            # deque(dataype, maxlen of deque), note [(0,0)] is a list of tuples
            self.channelBuffers.append(collections.deque([(0, 0)], self.historyLength))

        # This creates a list of buffers for FFT since FFT may require more data
        self.FFTBuffers = []
        for i in range(self.numberOfChannles):
            # deque(dataype, maxlen of deque), note [(0,0)] is a list of tuples
            self.FFTBuffers.append(collections.deque([(0, 0)], self.FFTLen))

        # TESTING, THIS WILL BE REMOVED LATER
        self.file = FileHandling("01", self.numberOfChannles)

    # This sets up the daq task and then spins out a process and thread to read from the DAQ
    def startUpdatingChannels(self):
        self.queues = []
        self.queues.append(queue.Queue())
        self.queues.append(queue.Queue())

        # Sets up a process to collect from the daq
        self.queues = []
        for i in range(self.numberOfChannles):
            self.queues.append(multiprocessing.Queue(maxsize=2*self.historyLength))

        self.process = multiprocessing.Process(target=NIDAQ.readFromDaqContinuosly,
                args=(self.numberOfChannles, self.device, self.samplingRate,
                        self.minVal, self.maxVal, self.sampsAtATime, self.queues))
        self.process.start()

        # Starts a thread to read from the DAQ process
        self.DAQThread = Thread(target=self.readIntoBuffersContinuosly)
        self.DAQThread.start()

    # Stops the recording process by killing the thread and the process
    def stopUpdatingChannels(self):
        # The thread must be terminated first or the queue.put waits for more items
        self.DAQThread.continueRunning = False
        time.sleep(0.05)
        self.DAQThread.join()

        self.process.terminate()
        time.sleep(0.05)
        self.process.join()

    # Must be called called on an individual thread to handle constant updating
    def readIntoBuffersContinuosly(self):
        # This allows the thread to be stopped from the function that called it
        NIDAQThread = threading.currentThread()
        while getattr(NIDAQThread, "continueRunning", True):

            channelValues = []
            for i in range(self.numberOfChannles):
                    daqValue = self.queues[i].get()
                    channelValues.append(daqValue)
                    self.channelBuffers[i].append((self.timeElapsed,daqValue))
                    self.FFTBuffers[i].append((self.timeElapsed, daqValue))

            #start = time.process_time()
            self.file.SaveData(self.timeElapsed, channelValues)
            #print("file Time: " + str(time.process_time() - start), flush=True)

            self.timeElapsed += (1/self.samplingRate)
        self.file.Close()

    # Must be called called on an individual process to handle constant updating
    @staticmethod
    def readFromDaqContinuosly(numChannels, device, samplingRate, minVal, maxVal, sampsAtATime, queues):
        # Create a new task to perform the reading, this task will die when this method ends
        with nidaqmx.Task() as readTask:
            # Add all of the channels up to the self.number of channels
            for i in range(numChannels):
                # RSE = reference single ended
                readTask.ai_channels.add_ai_voltage_chan(device + "ai" + str(i),
                    max_val=maxVal, min_val=minVal,
                    terminal_config=TerminalConfiguration.RSE)

            # This ensures that the DAQ is constantly sampling without prompt
            readTask.timing.cfg_samp_clk_timing(samplingRate,
                sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=samplingRate)
            readTask.start()

            # Stream reading allows for more elegant acquition at high rates
            reader = AnalogMultiChannelReader(readTask.in_stream)

            # Streamreader requires a numpy array to save values to
            holderArray = numpy.zeros((numChannels, sampsAtATime),
                dtype=numpy.float64)

            # Must be terminated by the parent process
            while (True):
                # Returns the number of samples read (same for each channel)
                # Waits until sampsAtATime number of samples are availabe
                reader.read_many_sample(holderArray, number_of_samples_per_channel=sampsAtATime)

                # Append read values into the queues for each channel to be read
                # by the parent process
                for i in range(len(queues)):
                    for j in range(sampsAtATime):
                        queues[i].put(holderArray[i][j])

# Responsible for displaying and updating the data to graph
class GraphValues(Widget):
    time_graph = ObjectProperty(None)
    frequency_graph = ObjectProperty(None)
    xMin = NumericProperty(0)
    xMax = NumericProperty(1)

    # Init must take in *kwargs for some reason. Something to do with inheriting from Widget
    def __init__(self, NIDevice, **kwargs):
        super(GraphValues, self).__init__(**kwargs)
        self.channelsToPlot = [0, 1, 2]
        self.DAQSampleRate = 1000
        self.histLen = 4000

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
        # Disable the start button to avoid trying to reserve the resource again
        self.ids.start_button.disabled = True
        self.ids.stop_button.disabled = False

        # This reseres the DAQ and it continusly gathers voltages in the buffers
        self.DAQ.startUpdatingChannels()

        # This gets the graph to update every 0.02 seconds
        Clock.schedule_interval(self.updateGraph, 0.05)

    def stop(self):
        # Re-enable the start Button, disable stop button for aesthetics
        self.ids.start_button.disabled = False
        self.ids.stop_button.disabled = True

        # This stops the loop updating the buffer and waits for the thread to finish
        self.DAQ.stopUpdatingChannels()

        # This stops the graph from updating
        Clock.unschedule(self.updateGraph)

    # dt is update time interval and must be passed to any funciton called from clock
    def updateGraph(self, dt):
        #ffts = FFT.FFTFromDEQ(self.DAQ.channelBuffers, len(self.channelsToPlot), self.DAQSampleRate)

        # Update the x boundaries to "follow" the graph
        self.xMin = self.DAQ.channelBuffers[0][0][0]
        self.xMax = self.DAQ.channelBuffers[0][0][0] + (self.histLen/self.DAQSampleRate)

        for timePlot, timeVals in zip(self.timePlots, self.DAQ.channelBuffers):
            timePlot.points = timeVals

        '''for timePlot, timeVals, freqPlot, freqVals in zip(self.timePlots,
                self.DAQ.channelBuffers, self.freqPlots, ffts):
            timePlot.points = timeVals
            freqPlot.points = freqVals'''

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
