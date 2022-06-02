from gtts import gTTS # pip install gtts
from IPython.display import Audio
from IPython.core.display import display

def speak(text):
    tts = gTTS(text = text, lang = 'ko')
    filename = 'warning.mp3'
    tts.save(filename)
    display(Audio('warning.mp3', autoplay=True))

if __name__ == '__main__':
    speak()