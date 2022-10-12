import whisper
import sounddevice as sd
import numpy as np
import streamlit as st
from scipy.io.wavfile import write
import soundfile as sf

if st.button('Record'):
    st.write("⏺Recording Audio..")
    fs = 44100
    duration = 10  # seconds
    my_recording = sd.rec(duration * fs, samplerate=fs, channels=2, dtype='float64')
    sd.wait()
    st.write('recording stopped!')

# TODO: How to use the stop button after the recording has started
    if st.button('stop'):
        sd.stop()
        st.write('recording stopped!')

# TODO: How to not save the file first
        # writing the audio to a .wav file
        write('output1.wav', fs, my_recording)


if st.button('Transcribe my Recording'):
    st.write('Model is listening..')

    # passing saved audio to whisper
    model = whisper.load_model("base")
    result = model.transcribe('output1.wav', fp16=False)
    st.write(result["text"])

# Option to play audio
st.write('Click here to listen to your recording')
if st.button('Play my recording'):
    data, fp = sf.read('output1.wav', dtype='float32')
    sd.play(data, fp)
    status = sd.wait()
