"""Real time plotting of Microphone level using kivy
"""

from kivy.lang import Builder
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.garden.graph import MeshLinePlot, Graph
from kivy.clock import Clock

from threading import Thread
import time
import collections

import nidaqmx
from nidaqmx.stream_readers import (AnalogSingleChannelReader, AnalogMultiChannelReader)
from nidaqmx.constants import (AcquisitionType, CountDirection, Edge,
    READ_ALL_AVAILABLE, TaskMode, TriggerType)
#import audioop
#import pyaudio

def get_microphone_level():
    print("Getting vals...")
    with nidaqmx.Task() as read_task:
        read_task.ai_channels.add_ai_voltage_chan("Dev1/ai0", max_val=1, min_val=-1)
        #read_task.ai_channels.add_ai_voltage_chan("Dev1/ai1", max_val=1, min_val=-1)

        sampling_rate = 500
        read_task.timing.cfg_samp_clk_timing(sampling_rate,
                                        sample_mode=AcquisitionType.CONTINUOUS)

        reader = AnalogSingleChannelReader(read_task.in_stream)

        value_read = reader.read_one_sample()

        print("value read: " + str(value_read))

        global levels
        #global levels2
        while True:
            time.sleep(0.001)
            #if len(levels) >= 100:
                #levels = []
            value_read = reader.read_one_sample()
            levels.append(value_read)
            #levels2.append(value_read)

class Logic(BoxLayout):
    def __init__(self, **kwargs):
        super(Logic, self).__init__(**kwargs)
        self.graph = Graph(xlabel='X', ylabel='Y', x_ticks_minor=5,
                x_ticks_major=25, y_ticks_major=1,
                y_grid_label=True, x_grid_label=True, padding=5,
                x_grid=True, y_grid=True, xmin=-0, xmax=500, ymin=-1, ymax=1)
        self.plot = MeshLinePlot(color=[0, 1, 1, 1])
        #self.plot2 = MeshLinePlot(color=[1, 0, 0, 1])

        self.add_widget(self.graph)

    def start(self):
        self.graph.add_plot(self.plot)
        Clock.schedule_interval(self.get_value, 0.002)

    def stop(self):
        Clock.unschedule(self.get_value)

    def get_value(self, dt):
        self.plot.points = [(i, j/5) for i, j in enumerate(levels)]
        #self.plot2.points = [(i, j/5) for i, j in enumerate(levels2)]
        #print(levels)


class RealTimeMicrophone(App):
    def build(self):
        return Builder.load_file("look.kv")

if __name__ == "__main__":
    levels = collections.deque(2500*[0], 2500)  # store levels of microphone
    levels2 = collections.deque(2500*[0], 2500)
    get_level_thread = Thread(target = get_microphone_level)
    get_level_thread.daemon = True
    get_level_thread.start()
    RealTimeMicrophone().run()
