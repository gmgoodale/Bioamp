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
        read_task.ai_channels.add_ai_voltage_chan(
            "Dev1/ai0", max_val=10, min_val=-10)

        sampling_rate = 2
        read_task.timing.cfg_samp_clk_timing(sampling_rate,
                                        sample_mode=AcquisitionType.CONTINUOUS)

        reader = AnalogSingleChannelReader(read_task.in_stream)

        # Generate random values to test.
        values_to_test = [random.uniform(-10, 10) for _ in range(10)]

        #values_read = []
        values_read = collections.deque(5*[0], 5)
        #number_of_samples = 4
        for i in range(100):
            value_read = reader.read_one_sample()
            values_read.append(value_read)
            #print(value_read)
            #read_array = numpy.zeros(number_of_samples, dtype=numpy.float64)
            #reader.read_many_sample(read_array, number_of_samples_per_channel=number_of_samples, timeout=10.0)
            print(values_read)
            #values_read.append(value_read)
            time.sleep(2)
