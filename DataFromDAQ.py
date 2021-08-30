# NI DAQmx for windows must be installed to use nidaqmx
import nidaqmx
import time

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0")

    while(True):
        print(task.read())
        time.sleep(1)
