import collections

channelBuffers = []
for i in range(5):
    channelBuffers.append(collections.deque(5*[0], 5))

print(channelBuffers)

channelBuffers[1].append(4)

print(channelBuffers)
