
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

labels=[]


all_wave = []
all_label = []
pathlist=[]


paths=[r'C:\Users\Lenovo\Desktop\project music\songdataem\Crema']
output_class=['ANG','DIS','FEA','HAP','NEU','SAD']

for train_audio_path in paths:
    ff=os.listdir(train_audio_path)



    files1 = os.listdir(train_audio_path)
    tf=train_audio_path.split("\\")
    print(tf)

    for wav in files1:
        if not wav.endswith(".wav"): continue
        print(train_audio_path+"\\" + wav)
        wave, sr = librosa.load(train_audio_path+"\\" + wav, mono=True)
        # samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 6500)

        # samples = librosa.resample(samples, sample_rate, 6500)
        mfcc = librosa.feature.mfcc(wave, sr)
        if len(mfcc[0])<80:
            mfcc = np.pad(mfcc[0], (0, (80 - len(mfcc[0]))), mode='constant', constant_values=0)
        samples=[]
        for i in range(0,20):
            for j in range(80):
                try:
                    samples.append(mfcc[i][j])
                except:
                    samples.append(0)
        if(len(samples)== 80*20) :

            all_wave.append(samples)
            label = 0
            for ll in range(len(output_class)):
                if output_class[ll] in wav:
                    label = ll
                    break
            all_label.append(ll)
        # if len(all_label)>300:
        #     break
print(len(all_label))
data=[]
for i in range(len(all_wave)):
    data.append({"label":all_label[i],"wav":all_wave[i]})
print(labels)
#
import csv

# Example data (assuming all_label and all_wave are lists of equal length)
data = [{"label": all_label[i], "wav": all_wave[i] } for i in range(len(all_label))]

# Define CSV file name
csv_filename = "output.csv"

# Write to CSV file
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["label", "wav",'path'])
    writer.writeheader()
    writer.writerows(data)

print(f"Data saved to {csv_filename}")

