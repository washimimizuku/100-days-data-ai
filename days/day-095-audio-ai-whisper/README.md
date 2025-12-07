# Day 95: Audio AI with Whisper

## Learning Objectives

**Time**: 1 hour

- Understand audio processing fundamentals
- Learn speech recognition with Whisper
- Implement audio transcription and translation
- Apply audio analysis techniques

## Theory (15 minutes)

### What is Audio AI?

Audio AI enables machines to process, understand, and generate audio signals including speech, music, and environmental sounds.

**Key Tasks**:
- Speech recognition
- Audio transcription
- Speaker identification
- Audio classification
- Text-to-speech
- Audio generation

### Audio Basics

**Audio Representation**:
```python
import numpy as np

# Audio is a time series of amplitude values
# Sample rate: samples per second (e.g., 16000 Hz)
# Duration: length in seconds

sample_rate = 16000
duration = 3  # seconds
audio = np.random.randn(sample_rate * duration)
```

**Audio Formats**:
- WAV: Uncompressed, high quality
- MP3: Compressed, smaller size
- FLAC: Lossless compression
- OGG: Open-source compression

### OpenAI Whisper

**What is Whisper?**: State-of-the-art speech recognition model trained on 680,000 hours of multilingual data.

**Key Features**:
- Multilingual (99 languages)
- Robust to accents and noise
- Automatic language detection
- Timestamp generation
- Translation to English

**Model Sizes**:
- tiny: 39M parameters, fastest
- base: 74M parameters
- small: 244M parameters
- medium: 769M parameters
- large: 1550M parameters, most accurate

### Speech Recognition

**Basic Transcription**:
```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")

print(result["text"])
# "Hello, this is a test transcription."
```

**With Timestamps**:
```python
result = model.transcribe("audio.mp3", word_timestamps=True)

for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
```

### Language Detection

**Automatic Detection**:
```python
# Whisper automatically detects language
result = model.transcribe("audio.mp3")
print(f"Detected language: {result['language']}")
```

**Specify Language**:
```python
# Force specific language
result = model.transcribe("audio.mp3", language="es")
```

### Translation

**Translate to English**:
```python
# Transcribe and translate to English
result = model.transcribe("spanish_audio.mp3", task="translate")
print(result["text"])  # English translation
```

### Audio Preprocessing

**Load Audio**:
```python
import librosa

# Load audio file
audio, sr = librosa.load("audio.mp3", sr=16000)

# Get duration
duration = librosa.get_duration(y=audio, sr=sr)
```

**Resample**:
```python
# Resample to different rate
audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=8000)
```

**Trim Silence**:
```python
# Remove silence from beginning and end
audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
```

### Audio Features

**Mel Spectrogram**:
```python
# Convert to mel spectrogram
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)

# Convert to dB scale
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
```

**MFCCs** (Mel-frequency cepstral coefficients):
```python
# Extract MFCCs
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
```

### Speaker Diarization

**Identify Speakers**:
```python
# Separate audio by speaker
# (Requires additional libraries like pyannote.audio)

from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
diarization = pipeline("audio.wav")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker {speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
```

### Audio Classification

**Classify Sounds**:
```python
from transformers import pipeline

classifier = pipeline("audio-classification", 
                     model="MIT/ast-finetuned-audioset-10-10-0.4593")

result = classifier("audio.wav")
# [{'label': 'Speech', 'score': 0.95}]
```

### Text-to-Speech

**Generate Speech**:
```python
from gtts import gTTS

text = "Hello, this is a test."
tts = gTTS(text=text, lang='en')
tts.save("output.mp3")
```

### Real-time Transcription

**Stream Audio**:
```python
import pyaudio
import wave

# Record audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

frames = []
for _ in range(0, int(RATE / CHUNK * 5)):  # 5 seconds
    data = stream.read(CHUNK)
    frames.append(data)

# Save and transcribe
wf = wave.open("recording.wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

result = model.transcribe("recording.wav")
```

### Evaluation Metrics

**Word Error Rate (WER)**:
```python
from jiwer import wer

reference = "hello world"
hypothesis = "hello word"

error_rate = wer(reference, hypothesis)
# 0.5 (50% error rate)
```

**Character Error Rate (CER)**:
```python
from jiwer import cer

error_rate = cer(reference, hypothesis)
```

### Use Cases

**Accessibility**:
- Closed captioning
- Voice commands
- Audio descriptions

**Content Creation**:
- Podcast transcription
- Video subtitles
- Meeting notes

**Customer Service**:
- Call center analytics
- Voice assistants
- Sentiment analysis

**Healthcare**:
- Medical dictation
- Patient interviews
- Clinical notes

### Best Practices

1. **Audio Quality**: Use high-quality recordings
2. **Preprocessing**: Remove noise and normalize
3. **Model Selection**: Balance accuracy and speed
4. **Language**: Specify when known
5. **Post-processing**: Clean and format output
6. **Validation**: Check transcription accuracy

### Why This Matters

Audio AI enables voice interfaces, accessibility features, and automated transcription. Whisper democratizes speech recognition with state-of-the-art accuracy across multiple languages, making audio AI accessible to developers.

## Exercise (40 minutes)

Complete the exercises in `exercise.py`:

1. **Audio Loading**: Load and process audio files
2. **Transcription**: Transcribe speech to text
3. **Language Detection**: Detect audio language
4. **Translation**: Translate speech to English
5. **Feature Extraction**: Extract audio features

## Resources

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Librosa Documentation](https://librosa.org/)
- [PyAudio Documentation](https://people.csail.mit.edu/hubert/pyaudio/)
- [Speech Recognition Guide](https://realpython.com/python-speech-recognition/)

## Next Steps

- Complete the exercises
- Review the solution
- Take the quiz
- Move to Day 96: Reinforcement Learning

Tomorrow you'll learn about reinforcement learning including agents, environments, rewards, and training RL models.
