import streamlit as st
import whisper
import numpy as np
#from pydub import AudioSegment


def run_uploads():
    audio_input = st.file_uploader("Upload audio here:", type = ["mp3","m4a","mp4","wav"])

    if audio_input is not None:
        audio_ = audio_input.read()

    output = ""
    with st.spinner("Transcribing in progress..."):
        if st.button("Transcribe"):
            model = whisper.load_model("base")
            result = model.transcribe(audio_)
            output = result["text"]

    return st.text_area("",output, placeholder = "Transcribed text...")

if __name__ == "__main__":
    run_uploads()