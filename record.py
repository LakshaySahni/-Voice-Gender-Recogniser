import wave
from scipy.io import wavfile
import sounddevice as sd
import numpy as np
from rpy2.robjects.conversion import py2ro, ri2py
from rpy2.robjects.packages import importr
from rpy2.robjects import r
from rpy2.robjects import numpy2ri
from prediction import Prediction
from pandas.rpy import common
numpy2ri.activate()
seewave = importr("seewave")
tuner = importr("tuneR")


def get_audio():
    duration = 3
    sample_rate = 44100
    print "Recording..."
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                       channels=2, blocking=True)
    print "Recorded output.wav for", str(duration), "seconds @", str(sample_rate), "kHz"
    wavfile.write('./test.wav', rate=44100, data=recording)
    sd.play(recording)
    return recording


def vectorize_audio():
    # get_audio()
    audio = tuner.readWave('./test2.wav')
    spec_object = seewave.spec(audio, 44100, plot=False)
    f = seewave.specprop(spec_object)
    r_features = common.convert_robj(f)
    fundamental_frequencies = seewave.fund(audio, 44100, plot=False)
    fundamental_frequencies = np.array(
        common.convert_robj(fundamental_frequencies).iloc[:, 0])
    fundamental_frequencies = fundamental_frequencies * 1000
    dominant_frequencies = seewave.dfreq(audio, 44100, plot=False)
    dominant_frequencies = np.array(
        common.convert_robj(dominant_frequencies).iloc[:, 0])
    dominant_frequencies = dominant_frequencies * 1000
    features = []
    features.append(r_features['mean'][0])
    features.append(r_features['sd'][0])
    features.append(r_features['median'][0])
    features.append(r_features['Q25'][0])
    features.append(r_features['Q75'][0])
    features.append(r_features['IQR'][0])
    features.append(r_features['skewness'][0])
    features.append(r_features['kurtosis'][0])
    features.append(r_features['sh'][0])
    features.append(r_features['sfm'][0])
    features.append(r_features['mode'][0])
    features.append(r_features['cent'][0])
    features.append(np.mean(fundamental_frequencies))
    features.append(np.min(fundamental_frequencies))
    features.append(np.max(fundamental_frequencies))
    features.append(np.mean(dominant_frequencies))
    features.append(np.min(dominant_frequencies))
    features.append(np.max(dominant_frequencies))
    features.append(np.max(dominant_frequencies) -
                    np.min(dominant_frequencies))
    features.append(0)
    return np.array(features)

prediction = Prediction()
prediction.predict(vectorize_audio())
