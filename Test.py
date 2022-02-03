import collections
import numpy
from scipy.fftpack import fft

def FFTFromDEQ(timeDEQs, numChannels, samplingRate):
      # Convert to a numpy array and take the fft for each channel
      chFrequencies = []
      for DEQ in timeDEQs:
          timeArray = numpy.asarray(DEQ)[:,1]
          chFrequencies.append(numpy.abs(fft(timeArray)))

      print(chFrequencies[0])

      for thing in chFrequencies[0]:
          print(thing)

      # Generate the X axis values which are the discrete frequency values
      N = len(chFrequencies[0])
      n = numpy.arange(N)
      T = N/samplingRate
      freqValues = n/T

      print(freqValues)

      freqGraphs = []
      # Generate a list of tuples from the X and Y values for graphing
      for ch in chFrequencies:
          freqGraph = []
          for x, y in zip(freqValues, ch):
              freqGraph.append((x,y))
          print(freqGraph)

          freqGraphs.append(freqGraph)

      return freqGraphs

if __name__ == "__main__":
    timeDEQs = []
    timeDEQs.append(collections.deque([(0, 1.1), (1, 1.2), (2, 1.3), (3, 1.4)]))
    timeDEQs.append(collections.deque([(0, 2.1), (1, 2.2), (2, 2.3), (3, 2.4)]))

    FFTFromDEQ(timeDEQs, 2, 2)
