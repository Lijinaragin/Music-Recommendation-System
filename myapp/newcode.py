from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
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
import pickle

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
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

warnings.filterwarnings("ignore")

path = r"C:\Users\Lenovo\Desktop\project music\songdataem\Crema/"
output_class=['ANG','DIS','FEA','HAP','NEU','SAD',]
all_wave = []
all_label = []



files1 = os.listdir(path)
for wav in files1:
            if not wav.endswith(".wav"): continue
            wave, sr = librosa.load(path+"\\" + wav, mono=True)
            # samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 6500)

            # samples = librosa.resample(samples, sample_rate, 6500)
            mfcc = librosa.feature.mfcc(wave, sr)
            # samples = np.pad(samples, (0, (6500 - len(samples))), mode='constant', constant_values=0)
            # print (len(mfcc[0]),type(mfcc[0]),mfcc[0])
            if len(mfcc[0])<80:
                mfcc = np.pad(mfcc, ((0, 0), (0, 80 - len(mfcc[0]))), mode='constant', constant_values=0)

            # samples=np.pad(mfcc[0], (0, (15 - len(mfcc[0]))), mode='constant', constant_values=0)
            samples=[]
            for i in range(0,20):
                for j in range(80):
                    samples.append(mfcc[i][j])
            if(len(samples)== 80*20) :
                label = 0
                for i in range(len(output_class)):
                    if output_class[i] in wav:
                        label = i
                        break
                all_wave.append(list(samples))
                all_label.append(int(label))

print (all_label)
X=np.array(all_wave)
Y=all_label
# X,Y=loadDataSet()
# print (len(X[1]),"yyyyyyyyyyy",X[1])
model = Sequential()
model.add(Dense(12, input_dim=1600, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.10, epochs=20, batch_size=10, verbose=0)
print (history)
model.save("emotion_classification_model.h5")



yyy=[X[90]]
yyy=numpy.array(yyy)

res=model.predict_classes(yyy,batch_size=1, verbose=0)
print(res)
print (Y[90])
# # list all data in history
# print(history.history.keys())
#
print(history.history['accuracy'])