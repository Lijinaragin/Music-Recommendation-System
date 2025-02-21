import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
x=[]
y=[]

resultlist=[]
with open(r"C:\Users\Lenovo\PycharmProjects\music\myapp\output.csv", mode="r") as file:
    reader = csv.DictReader(file)
    kk=0
    for row in reader:
        # print(row['label'])
        # print(row['wav'],type(row['wav']))
        xx=row['wav'].replace("[","").replace("]","").split(",")
        xy=[]
        for i in xx:
            xy.append(float(i))
        x.append(xy)
        resultlist.append(row)
        y.append(int(row['label']))


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create KNN model with k=1
knn = KNeighborsClassifier(n_neighbors=1)
print(x[0])
print(len(x[0]))
# Train the model
knn.fit(x, y)


import librosa

import numpy as np


import os
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
    res=knn.predict(all_wave)
    print(res)
    x=resultlist[int(res)]
    print(x['wav'])


    print(x['label'])



    print(res)

predict_audio(r"1001_ITS_SAD_XX.wav")