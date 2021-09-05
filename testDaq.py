import nidaqmx
from nidaqmx.constants import (AcquisitionType, CountDirection, Edge,
    READ_ALL_AVAILABLE, TaskMode, TriggerType)
from nidaqmx.stream_readers import CounterReader
import numpy

# Let's load up the NI-DAQmx system that is visible in the
# Measurement & Automation Explorer (MAX) software of NI-DAQmx for
# the local machine.
system = nidaqmx.system.System.local()
# We know on our current system that our DAQ is named 'DAQ1'
DAQ_device = system.devices['Dev1']
# create a list of all the counters available on 'DAQ1'
counter_names = [ci.name for ci in DAQ_device.ci_physical_chans]
print(counter_names)
# note that using the counter output channels instead of the inputs
# includes the '[device]/freqout' output, which is not a counter
print([co.name for co in DAQ_device.co_physical_chans])

with nidaqmx.Task() as read_task:
    # create a digital input channel on 'port0' of 'DAQ1'
    read_task.ai_channels.add_ai_voltage_chan("Dev1/ai0")

    # configure the timing parameters of the sample clock so that
    # it has a sampling rate of 100 Hz and runs continuously so
    # that the digital input sample clock continues to run even if
    # it's associated task is not reading anything from the channel.
    sampling_rate = 100
    read_task.timing.cfg_samp_clk_timing(sampling_rate,
                                    sample_mode=AcquisitionType.CONTINUOUS)
    # commit the task from the Reserved state in system memory to
    # the Commit state on the DAQ; this programs the hardware
    # resources with those settings of the task which must be
    # configured before the task transitions into the Start state.
    # This speeds up the execution of the samp_clk_task.start() call
    # because the hardware will now be in the Commit state and must
    # only transition to the State state to run the task.
    read_task.control(TaskMode.TASK_COMMIT)


    # set the buffer size of the counter, such that, given the
    # sampling rate at which the counter reads out its current value
    # to the buffer, it will give two minutes of samples before the
    # buffer overflows.
    read_task.in_stream.input_buf_size = 12000

    reader = CounterReader(read_task.in_stream)
    # start the tasks to begin data acquisition; note that because
    # the arm start trigger of the counter was set, it does not
    # matter which task is started first, the tasks will be synced
    read_task.start()
    # create a data buffer for the counter stream reader
    data_array = numpy.zeros(0, dtype=numpy.uint32)
    # read all samples from the counter buffer to the system memory
    # buffer data_array; if the buffer is not large enough, it will
    # raise an error
    reader.read_many_sample_uint32(data_array,
        number_of_samples_per_channel=READ_ALL_AVAILABLE)

    print(data_array)
