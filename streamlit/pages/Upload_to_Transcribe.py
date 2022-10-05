import streamlit as st
import whisper
# import numpy as np
import torchaudio
from io import BytesIO
#from pydub import AudioSegment


def run_uploads():
    audio_input = st.file_uploader("Upload audio here:", type = ["mp3","m4a","mp4","wav"])

    if audio_input is not None:
        audio_ = BytesIO(audio_input.read())
        waveform, _ = torchaudio.load(audio_)

    output = ""
    with st.spinner("Transcribing in progress..."):
        if st.button("Transcribe"):
            model = whisper.load_model("base")
            # result = model.transcribe(waveform)
            # output = result["text"]
            audio = whisper.pad_or_trim(waveform.flatten()).to(model.device)
            mel = whisper.log_mel_spectrogram(audio)
            
            # decode the audio
            options = whisper.DecodingOptions(fp16 = False,language="en", without_timestamps=True)
            result = model.decode(mel, options)
            output = result.text

    return st.text_area("",output, placeholder = "Transcribed text...")

if __name__ == "__main__":
    run_uploads()