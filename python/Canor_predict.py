



import os
import time
import random
import numpy as np
import librosa
from keras.models import load_model
import tensorflow as tf
from tensorflow.python.platform import gfile


n_fft =1024
hop_length = 512
song_samples = 660000

def splitsongs(X, window = 0.1, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
        
    return np.array(temp_X)

def to_melspectrogram(songs, n_fft = 1024, hop_length = 512):
    
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft = n_fft,
        hop_length = hop_length)[:,:,np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    return np.array(list(tsongs))

def audio_preprocess(path):
    X =[]
    waves = gfile.Glob(path)
  
    for wave_path in waves:
        signal, sr = librosa.load(wave_path)
        signal = signal[:song_samples]
        signals = splitsongs(signal)
        specs = to_melspectrogram(signals, n_fft, hop_length)
      
        X.extend(specs)
    return np.array(X)

def main():

  #provide path of the song
    song_path = "./Hey_Sailor.mp3"
    audio_data = audio_preprocess(song_path)

  #provide the H5 model input size (?,128,129,1)
    model = load_model('./Canor_model.h5')

    list_pred = model.predict_classes(audio_data)
    Max_pred = np.bincount(list_pred)
    Pred_genre= np.argmax(Max_pred)
    labels = ['jazz', 'metal', 'hiphop', 'classical', 'disco', 'blues', 'reggae', 'pop', 'country', 'rock']
    result = labels[Pred_genre]
    print(result)

if __name__=='__main__':
    main()