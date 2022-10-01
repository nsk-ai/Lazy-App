# Lazy-App

## Setup
Fork the repo  
Run these commands to setup the required packages for the model
```bash
pip install git+https://github.com/openai/whisper.git 
```

It also requires the command-line tool ffmpeg to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

## Tasks
- [ ] Run the model locally first
- [ ] Streamlit  
  - [ ] Create Homepage integrate NSK.ai logo
  - [ ] Integrate python (low-level API) with streamlit
  - [ ] Accept audio input with streamlit audio input
  - [ ] Option of uploading MP3 input
  - [ ] (optional) [Real-time audio translation](https://towardsdatascience.com/how-to-transcribe-streams-of-audio-data-in-real-time-with-python-and-assemblyai-322da8b5b7c9)

## Python Usage
I recommend we use the low-level API for our app

```python
# High-level API
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])
```

```python
import whisper

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
```

For more information, Look at the [whisper](https://github.com/openai/whisper) documentation