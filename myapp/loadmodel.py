from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
from keras.models import load_model
import os
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import warnings
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
# load pima indians dataset
from tensorflow.python.keras.distribute.keras_correctness_test_base import get_batch_size

# dataset = numpy.loadtxt("innovators.csv", delimiter=",")
# split into input (X) and output (Y) variables
import numpy as np
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)
model=load_model('my_trained_model.h5')


def predict_audio(file):
    wave, sr = librosa.load(file, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    all_wave=[]
    if len(mfcc[0]) < 80:
        mfcc = np.pad(mfcc[0], (0, (80 - len(mfcc[0]))), mode='constant', constant_values=0)
    samples = []
    for i in range(0, 20):
        for j in range(80):
            try:
                samples.append(mfcc[i][j])
            except:
                samples.append(0)
    if (len(samples) == 80 * 20):

        all_wave.append(samples)
    all_wave=np.array(all_wave)
    res=model.predict(all_wave)
    max_index = np.argmax(res[0])
    print(max_index)


predict_audio(r"ss.wav")