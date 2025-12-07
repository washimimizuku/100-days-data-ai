"""Day 95: Audio AI with Whisper - Solutions

NOTE: Mock implementations for learning without audio files.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import random


# Exercise 1: Audio Loading
class AudioLoader:
    """Load and process audio files."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        # Mock audio loading
        duration = random.uniform(2, 10)
        audio = np.random.randn(int(self.sample_rate * duration))
        return audio, self.sample_rate
    
    def get_duration(self, audio: np.ndarray) -> float:
        """Get audio duration in seconds."""
        return len(audio) / self.sample_rate
    
    def trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove silence from audio."""
        # Find non-silent regions
        mask = np.abs(audio) > threshold
        if not mask.any():
            return audio
        
        # Find first and last non-silent samples
        indices = np.where(mask)[0]
        start, end = indices[0], indices[-1] + 1
        
        return audio[start:end]


# Exercise 2: Speech Transcription
class SpeechTranscriber:
    """Transcribe speech to text."""
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.sample_texts = [
            "Hello, how are you today?",
            "This is a test transcription.",
            "Welcome to audio AI with Whisper."
        ]
    
    def transcribe(self, audio: np.ndarray) -> Dict[str, Any]:
        """Transcribe audio to text."""
        # Mock transcription
        text = random.choice(self.sample_texts)
        
        return {
            'text': text,
            'language': 'en',
            'segments': [
                {
                    'start': 0.0,
                    'end': len(audio) / 16000,
                    'text': text
                }
            ]
        }
    
    def transcribe_with_timestamps(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Transcribe with word-level timestamps."""
        text = random.choice(self.sample_texts)
        words = text.split()
        
        segments = []
        duration = len(audio) / 16000
        time_per_word = duration / len(words)
        
        for i, word in enumerate(words):
            segments.append({
                'word': word,
                'start': i * time_per_word,
                'end': (i + 1) * time_per_word
            })
        
        return segments


# Exercise 3: Language Detection
class LanguageDetector:
    """Detect audio language."""
    
    def __init__(self):
        self.supported_languages = ["en", "es", "fr", "de", "zh", "ja"]
    
    def detect_language(self, audio: np.ndarray) -> Dict[str, Any]:
        """Detect language from audio."""
        # Mock language detection
        language = random.choice(self.supported_languages)
        confidence = random.uniform(0.85, 0.99)
        
        return {
            'language': language,
            'confidence': confidence
        }
    
    def detect_multiple(self, audio_segments: List[np.ndarray]) -> List[str]:
        """Detect language for multiple segments."""
        return [self.detect_language(seg)['language'] for seg in audio_segments]


# Exercise 4: Audio Translation
class AudioTranslator:
    """Translate speech to English."""
    
    def __init__(self):
        self.translations = {
            'es': ('Hola, ¿cómo estás?', 'Hello, how are you?'),
            'fr': ('Bonjour, comment allez-vous?', 'Hello, how are you?'),
            'de': ('Hallo, wie geht es dir?', 'Hello, how are you?')
        }
    
    def translate(self, audio: np.ndarray, source_lang: str = None) -> Dict[str, Any]:
        """Translate audio to English."""
        # Mock translation
        if source_lang is None:
            source_lang = random.choice(['es', 'fr', 'de'])
        
        original, translated = self.translations.get(
            source_lang,
            ('Original text', 'Translated text')
        )
        
        return {
            'original_text': original,
            'translated_text': translated,
            'source_language': source_lang
        }
    
    def translate_batch(self, audio_files: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Translate multiple audio files."""
        return [self.translate(audio) for audio in audio_files]


# Exercise 5: Audio Feature Extraction
class AudioFeatureExtractor:
    """Extract features from audio."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram."""
        # Mock mel spectrogram (time x frequency)
        n_frames = len(audio) // 512
        n_mels = 80
        mel_spec = np.random.randn(n_mels, n_frames)
        return mel_spec
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCCs."""
        # Mock MFCCs (coefficients x time)
        n_frames = len(audio) // 512
        mfccs = np.random.randn(n_mfcc, n_frames)
        return mfccs
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract multiple features."""
        return {
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'mfcc': self.extract_mfcc(audio),
            'duration': len(audio) / self.sample_rate,
            'sample_rate': self.sample_rate
        }


# Bonus: Speaker Diarization
class SpeakerDiarizer:
    """Identify different speakers."""
    
    def diarize(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Identify speakers in audio."""
        duration = len(audio) / 16000
        num_speakers = random.randint(1, 3)
        
        segments = []
        current_time = 0.0
        
        for i in range(random.randint(3, 6)):
            speaker = f"SPEAKER_{i % num_speakers:02d}"
            segment_duration = random.uniform(1, 3)
            
            if current_time + segment_duration > duration:
                segment_duration = duration - current_time
            
            segments.append({
                'speaker': speaker,
                'start': current_time,
                'end': current_time + segment_duration
            })
            
            current_time += segment_duration
            
            if current_time >= duration:
                break
        
        return segments
    
    def count_speakers(self, audio: np.ndarray) -> int:
        """Count number of speakers."""
        segments = self.diarize(audio)
        speakers = set(seg['speaker'] for seg in segments)
        return len(speakers)


def demo_audio_ai():
    """Demonstrate audio AI tasks."""
    print("Day 95: Audio AI with Whisper - Solutions Demo\n" + "=" * 60)
    
    # Create mock audio
    sample_rate = 16000
    duration = 5
    audio = np.random.randn(sample_rate * duration)
    
    print("\n1. Audio Loading")
    loader = AudioLoader()
    loaded_audio, sr = loader.load_audio("mock_audio.wav")
    audio_duration = loader.get_duration(loaded_audio)
    trimmed = loader.trim_silence(loaded_audio)
    print(f"   Loaded audio: {len(loaded_audio)} samples at {sr} Hz")
    print(f"   Duration: {audio_duration:.2f} seconds")
    print(f"   After trimming: {len(trimmed)} samples")
    
    print("\n2. Speech Transcription")
    transcriber = SpeechTranscriber()
    result = transcriber.transcribe(audio)
    print(f"   Transcription: '{result['text']}'")
    print(f"   Language: {result['language']}")
    
    timestamps = transcriber.transcribe_with_timestamps(audio)
    print(f"   Word timestamps: {len(timestamps)} words")
    if timestamps:
        print(f"   First word: '{timestamps[0]['word']}' at {timestamps[0]['start']:.2f}s")
    
    print("\n3. Language Detection")
    detector = LanguageDetector()
    lang_result = detector.detect_language(audio)
    print(f"   Detected language: {lang_result['language']}")
    print(f"   Confidence: {lang_result['confidence']:.2%}")
    
    print("\n4. Audio Translation")
    translator = AudioTranslator()
    translation = translator.translate(audio, source_lang='es')
    print(f"   Original: '{translation['original_text']}'")
    print(f"   Translated: '{translation['translated_text']}'")
    print(f"   Source language: {translation['source_language']}")
    
    print("\n5. Feature Extraction")
    extractor = AudioFeatureExtractor()
    features = extractor.extract_all_features(audio)
    print(f"   Mel spectrogram shape: {features['mel_spectrogram'].shape}")
    print(f"   MFCC shape: {features['mfcc'].shape}")
    print(f"   Duration: {features['duration']:.2f}s")
    
    print("\n6. Speaker Diarization")
    diarizer = SpeakerDiarizer()
    segments = diarizer.diarize(audio)
    num_speakers = diarizer.count_speakers(audio)
    print(f"   Number of speakers: {num_speakers}")
    print(f"   Number of segments: {len(segments)}")
    if segments:
        print(f"   First segment: {segments[0]['speaker']} "
              f"({segments[0]['start']:.1f}s - {segments[0]['end']:.1f}s)")
    
    print("\n" + "=" * 60)
    print("All audio AI tasks demonstrated!")


if __name__ == "__main__":
    demo_audio_ai()
