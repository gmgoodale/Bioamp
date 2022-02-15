import collections
import re

import numpy
import pytest
import random
import time

import nidaqmx
#from nidaqmx.constants import Edge
#from nidaqmx.utils import flatten_channel_string
from nidaqmx.stream_readers import (AnalogSingleChannelReader, AnalogMultiChannelReader)
#from nidaqmx.stream_writers import (AnalogSingleChannelWriter, AnalogMultiChannelWriter)
#from nidaqmx.tests.fixtures import x_series_device
#from nidaqmx.tests.helpers import generate_random_seed
#from nidaqmx.tests.test_read_write import TestDAQmxIOBase'''
from nidaqmx.constants import (AcquisitionType, CountDirection, Edge,
    READ_ALL_AVAILABLE, TaskMode, TriggerType)

with nidaqmx.Task() as read_task:
        read_task.ai_channels.add_ai_voltage_chan("Dev1/ai0", max_val=10, min_val=-10)
        read_task.ai_channels.add_ai_voltage_chan("Dev1/ai1", max_val=10, min_val=-10)

        sampling_rate = 5000
        read_task.timing.cfg_samp_clk_timing(sampling_rate,
                                        sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=10000)
        read_task.start()

        reader = AnalogMultiChannelReader(read_task.in_stream)
        values_read1 = collections.deque(5*[0], 5)
        values_read2 = collections.deque(5*[0], 5)
        holder_array = numpy.zeros((2,2), dtype=numpy.float64)
        for i in range(100*sampling_rate):
            samplesRead = reader.read_many_sample(holder_array, number_of_samples_per_channel=2)
            values_read1.append([holder_array[0][0], holder_array[0][1]])
            values_read2.append([holder_array[1][0], holder_array[1][1]])
            print(values_read1, flush=True)
            print(values_read2, flush=True)
            #read_array = numpy.zeros(number_of_samples, dtype=numpy.float64)
            #reader.read_many_sample(read_array, number_of_samples_per_channel=number_of_samples, timeout=10.0)
            #values_read.append(value_read)
