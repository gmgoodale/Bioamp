"""Real time plotting of Microphone level using kivy
"""

from kivy.lang import Builder
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.garden.graph import MeshLinePlot, Graph
from kivy.clock import Clock
from threading import Thread
import time

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

        sampling_rate = 100
        read_task.timing.cfg_samp_clk_timing(sampling_rate,
                                        sample_mode=AcquisitionType.CONTINUOUS)

        reader = AnalogSingleChannelReader(read_task.in_stream)

        value_read = reader.read_one_sample()

        print("value read: " + str(value_read))

        global levels
        while True:
            time.sleep(0.01)
            if len(levels) >= 100:
                levels = []
            levels.append(value_read + 5)

class Logic(BoxLayout):
    def __init__(self, **kwargs):
        super(Logic, self).__init__(**kwargs)
        self.graph1 = Graph(xlabel='X', ylabel='Y', x_ticks_minor=5,
            x_ticks_major=25, y_ticks_major=1,
            y_grid_label=True, x_grid_label=True, padding=5,
            x_grid=True, y_grid=True, xmin=-0, xmax=100, ymin=-1, ymax=1)
        self.plot = MeshLinePlot(color=[1, 0, 0, 1])

    def start(self):
        print("start")
        self.graph1.add_plot(self.plot)
        Clock.schedule_interval(self.get_value, 0.01)

    def stop(self):
        Clock.unschedule(self.get_value)

    def get_value(self, dt):
        self.plot.points = [(i, j/5) for i, j in enumerate(levels)]
        #print(levels)


class RealTimeMicrophone(App):
    def build(self):
        return Builder.load_file("look.kv")

if __name__ == "__main__":
    levels = []  # store levels of microphone
    get_level_thread = Thread(target = get_microphone_level)
    get_level_thread.daemon = True
    get_level_thread.start()
    RealTimeMicrophone().run()
