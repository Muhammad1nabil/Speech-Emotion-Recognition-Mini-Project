import librosa
import soundfile
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# I added more observed_emotions but the accuracy decreases, 
# I think because Im using MLPClassifier and its not the best fit for this
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']


def extract_feature(file_name, mfcc, chroma, mel):
    # reading sound files
    # extracting features using mfccs chroma and melspectrogram
    with soundfile.SoundFile(file_name) as file:
        result = np.array([])
        X = file.read(dtype='float32')
        sample_rate = file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        return result


def load_data(test_size=0.2):
    x, y = [], []
    # using glob to load data from all Actor_* folders and all soundfiles in each one of them
    for file in glob.glob('dataset\Actor_*\*.wav'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split('-')[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, True, True, True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(x, y, test_size=test_size, random_state=9)


x_train, x_test, y_train, y_test = load_data(0.25)

# creating Multi-layer Perceptron classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                      hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score = accuracy_score(y_test, y_pred)

print(score)
