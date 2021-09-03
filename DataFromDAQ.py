# NI DAQmx for windows must be installed to use nidaqmx from: https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html#409845
import nidaqmx
import time

class DAQ(port):
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0")

    while(True):
        print(task.read())
        time.sleep(1)
