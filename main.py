import matplotlib.pyplot as plt
import re
import speech_recognition as sr
import requests
import io
from pydub import AudioSegment
import numpy as np
import gdown
import librosa
import soundfile as sf
from jiwer import wer, cer
from IPython.display import Audio

#списки для сбора результатов
wer_data = []
cer_data = []



def load_file(url):
    """
    Загрузка записи на диск
    """
    url = url
    output = 'file_to_conwert.mp3'
    target_url = 'https://drive.google.com/uc?export=download&id='+(re.split(r'd/',(re.split(r'/v', url )[0]))[1])
    gdown.download(target_url, output)
    file_path = '/content/file_to_conwert.mp3'
    return file_path



def download_and_convert_audio(file_path):
    """
     Аудио сигнал (Audio Signal):
    конвертация MP3 в WAV
    """

    audio = AudioSegment.from_file(file_path, format="mp3")
    wav_io = io.BytesIO()
    audio.export(wav_io, format='wav')
    wav_io.seek(0)
    return wav_io

def add_noise(file_path):
    """
    Добывление шума к записи
    """
    audio = AudioSegment.from_file(file_path, format="mp3")
    wav_io = io.BytesIO()
    audio.export(wav_io, format='wav')
    wav_io.seek(0)

    audio_data, sample_rate = librosa.load(wav_io, sr=None)


    # Уровень шума
    noise_level = 0.02

    # Генерация белого шума
    white_noise = noise_level * np.random.normal(size=len(audio_data))

    # Наложение шума на исходный сигнал
    noise_data = audio_data + white_noise
    #сохранение файла с шумом
    sf.write("noized_audio.wav", noise_data, sample_rate)
    return"Шум добавлен"

def recognize_poetry(audio_file):
    """
    Извлечение признаков (Feature Extraction) + Акустическая модель (Acoustic Model)
    Распознавание поэтического текста
    Библиотека speech_recognition реализует блоки Извлечение признаков и Акустическую модель.
    """
    recognizer = sr.Recognizer()

    # Настройка параметров распознавания
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8  # Увеличенная пауза для поэтической речи

    with sr.AudioFile(audio_file) as source:
        print("Обработка аудио...")
        # Записываем аудио с настройкой уровня окружающего шума
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)

    try:
        # 🔍 Декодер (Decoder):
        # Распознавание с использованием Google Speech Recognition, который включает в себя алгоритм поиска по лучу
        text = recognizer.recognize_google(audio, language="ru-RU")
        return text
    except sr.UnknownValueError:
        return "Не удалось распознать речь"
    except sr.RequestError as e:
        return f"Ошибка сервиса распознавания: {e}"


def main(url):

    print("Загрузка аудиофайла...")
    file_path = load_file(url)
    wav_file = download_and_convert_audio(file_path)

    print("Распознавание речи...")
    recognized_text = recognize_poetry(wav_file)

    print("\nРаспознанное стихотворение:")

    print(recognized_text)

    print("Добавление шума к аудио")
    add_noise(file_path)

    print("Распознавание речи файла с шумом...")
    noised_text = recognize_poetry("/content/noized_audio.wav")


    print("\nРаспознанное стихотворение с шумом:")

    print(noised_text)

    print("\nМетрики оценки распознания (WER, CER)")

    wer_value = wer(recognized_text, noised_text)
    print(f"WER: {wer_value}")

    # Пример вычисления CER
    cer_value = cer(recognized_text, noised_text)
    print(f"CER: {cer_value}")
    return wer_value, cer_value

#Запускаем модель для проверки
wer_wal, cer_val = main('[FILE_LINK]')
wer_data.append(wer_wal)
cer_data.append(cer_val)
print('Оригинал')
wn = Audio('/content/file_to_conwert.mp3', autoplay=False)
display(wn)
print('Запись с шумом')
wn = Audio('/content/noized_audio.wav', autoplay=False)
display(wn)

#Выведем график оценок
samples = [[Track_numbers]]
plt.bar(samples, wer_data,width=0.5, align='edge', label='WER')
plt.bar(samples, cer_data,width=0.5, align='center', label='CER')
plt.xlabel('Номер записи')
plt.legend()
plt.show()