
from lasagne.layers import *
from fuel.datasets.youtube_audio import YouTubeAudio

data = YouTubeAudio('XqaJ2Ol5cC4')

stream = data.get_example_stream()

it = stream.get_epoch_iterator()

seq = next(it)

print seq[0].shape



for epoch in range(0,100):




