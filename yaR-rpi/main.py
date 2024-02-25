import os
import RPi.GPIO as GPIO
import time

import wave
import pyaudio
import threading
import pygame

from picamera2.encoders import H264Encoder
from picamera2 import Picamera2


from request_handler import upload_video_and_handle_response
# button code
button_pin = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(button_pin,GPIO.IN, pull_up_down=GPIO.PUD_UP)

# audio code
class AudioRecorder:
    def __init__(self, filename, channels=1, rate=44100, frames_per_buffer=1024):
        self.filename = filename
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.pyaudio_instance = pyaudio.PyAudio()
        self.frames = []

    def record(self):
        self.is_recording = True
        stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
        )
        while self.is_recording:
            self.frames.append(stream.read(self.frames_per_buffer))

        stream.stop_stream()
        stream.close()

    def stop(self):
        self.is_recording = False

    def save(self):
        with wave.open(self.filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio_instance.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(self.frames))
def play_audio(audio_path):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


#camera code
picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)
encoder = H264Encoder(bitrate=10000000)
video_filename = "test.mp4"

is_recording = False
try:
    while True:
        button_state = GPIO.input(button_pin)
        token = 'c3BhcnNoIGRvZXMgY29jYWluZQ'
        if button_state == False:
            if not is_recording:
                picam2.start_recording(encoder, video_filename)
                audio_filename = "audio.wav"
                audio_recorder = AudioRecorder(audio_filename)
                audio_thread = threading.Thread(target=audio_recorder.record)
                audio_thread.start()

                is_recording = True
                print("Recording started")
            else:
                # Stop video recording
                picam2.stop_recording()

                # Stop audio recording
                audio_recorder.stop()
                audio_thread.join()
                audio_recorder.save()

                is_recording = False
                print("Recording stopped")
                start_time = time.time()

                mp3_path = upload_video_and_handle_response(video_filename,audio_filename,token)
                print("Request sent")
                if mp3_path:
                    time_taken = time.time()-start_time
                    print("Time taken to process",time_taken)
                    play_audio(mp3_path)

            time.sleep(0.2)


        time.sleep(0.5)
except KeyboardInterrupt:
    if is_recording:
        picam2.stop_recording()
        audio_recorder.stop()
        audio_thread.join()
        audio_recorder.save()
    GPIO.cleanup()
