# TTS text_to_speech
# STT speech_to_text

from gtts import gTTS
from playsound import playsound

file_name = 'sample.mp3'

#text파일 열기
with open("./utils/sample.txt", 'r', encoding='utf8') as f:
    text = f.read()

#ENG
# text = 'can i help you?'
# tts_en = gTTS(text = text, lang='en')
# tts_en.save(file_name)

#KOR
# text = '가나다라마바사'
tts_ko = gTTS(text = text, lang='ko')
tts_ko.save(file_name)


playsound(file_name) # mp3 file play