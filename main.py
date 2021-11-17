# NI imports
import nidaqmx
from nidaqmx.stream_readers import (AnalogSingleChannelReader, AnalogMultiChannelReader)
from nidaqmx.constants import (AcquisitionType, CountDirection, Edge,
    READ_ALL_AVAILABLE, TaskMode, TriggerType)

# Kivy imports
from kivy.app import App
from kivy.garden.graph import MeshLinePlot, Graph
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.properties import (NumericProperty, ReferenceListProperty,
    ObjectProperty)

# General imports
import collections
import numpy
import time
from threading import Thread

# Responsible for getting data from the NI DAQ and storing it
class DAQ:

    def __init__(self, device):
        self.device = device
        self.samplingRate = 100  # Sampling rate in Hz
        self.historyLength = 5 # Number of samples in buffer to be displayed
        self.numberOfChannles = 1 # Assumes channels are 0 -> numberOfChannles
        self.minVal = -1 # Sets the range for the DAQ in volts, reducing range increases precision
        self.maxVal = 1

        # This creates a list of FIFO buffers of a fixed size, these buffers are how you access the channel data
        self.channelBuffers = []
        for i in range(self.numberOfChannles):
            self.channelBuffers.append(collections.deque(self.historyLength*[0], self.historyLength))

    # Typically called on an individual thread to handle constant updating
    def updateChannelsContinuosly(self):
        # Create a new task to perform the reading, this task will die when this method ends
        with nidaqmx.Task() as readTask:
            # Add all of the channels connected at the box level (5)
            for i in range(self.numberOfChannles):
                readTask.ai_channels.add_ai_voltage_chan(self.device + "ai" + str(i),
                    max_val=self.maxVal, min_val=self.minVal)

            # This ensures that the DAQ is constantly sampling without prompt
            readTask.timing.cfg_samp_clk_timing(self.samplingRate,
                sample_mode=AcquisitionType.CONTINUOUS)

            # Stream reading allows for more elegant acquition at high rates
            self.reader = AnalogMultiChannelReader(readTask.in_stream)

            # Streamreader requires a numpy array to save values to
            holder_array = numpy.zeros(self.numberOfChannles, dtype=numpy.float64)

            while True:
                self.reader.read_one_sample(holder_array)

                for i in range(self.numberOfChannles):
                    self.channelBuffers[i].append(holder_array[i])

                time.sleep(1/(2*self.samplingRate))
                #print(self.channelBuffers, flush=True)

# Responsible for displaying and updating the data to graph
class Graph(Widget):
    graph = ObjectProperty(None)
    channelsToPlot = [0, 1]

    def __init__(self):
        self.plots = []

        for ch in channelsToPlot:
            plot = MeshLinePlot(color=self.ColorGenerator(ch))
            self.plots.append(plot)
            self.graph.add_plot(plot)

    def UpdateGraph(self):
        pass

    # This just returns a unique color for the first 5 channels, if there is a better way to do this please do
    def ColorGenerator(self, ch):
        if (ch == 0):
            return [1, 0, 0, 1]
        elif (ch == 1):
            return [0, 1, 1, 1]
        elif (ch == 2):
            return [0, 1, 0, 1]
        elif (ch == 3):
            return [0, 0, 1, 1]
        elif (ch == 4):
            return [1, 0, 1, 1]
        else:
            return [1, 1, 1, 1]


class DAQApp(App):
    def build(self):
        return Builder.load_file("DAQ.kv")

if __name__ == '__main__':
    #DAQApp().run()
    NI6008 = DAQ("Dev1/")
    NI6008.samplingRate = 5
    NI6008.updateChannelsContinuosly()
