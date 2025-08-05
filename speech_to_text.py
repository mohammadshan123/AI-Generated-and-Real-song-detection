import osfi
import speech_recognition as sr
from pydub import AudioSegment

def convert_to_wav(input_audio_path):
    """Convert any audio file to WAV format."""
    wav_path = os.path.splitext(input_audio_path)[0] + ".wav"
    audio = AudioSegment.from_file(input_audio_path)
    audio.export(wav_path, format="wav")
    return wav_path

def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    
    # Convert to WAV
    wav_path = convert_to_wav(audio_path)

    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError:
        return "Speech Recognition service is unavailable"

