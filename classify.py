import argparse
import time
import os

from audio_classifier import AudioClassifier
from audio_classifier import AudioClassifierOptions

# specify filename of tflite model
model = "soundclassifier_with_metadata.tflite"

# Initialize the audio classification model
options = AudioClassifierOptions(
    num_threads=4,
    max_results=5,
    score_threshold=0)
classifier = AudioClassifier(model, options)

# Initialize the audio recorder and a Tensorflow tensor to store the audio input.
audio_record = classifier.create_audio_record()
tensor_audio = classifier.create_input_tensor_audio()

# We will be calculating the input time by looking at the model's input length 
# Then we devide it by 2 to make it more accurate
input_length_in_second = float(len(
    tensor_audio.buffer)) / tensor_audio.format.sample_rate
interval_between_inference = input_length_in_second * 0.5
pause_time = interval_between_inference * 0.1
last_inference_time = time.time()

# Start audio recording in the background.
audio_record.start_recording()

# Loop forever and print putput
while True:
  # Wait until at least interval_between_inference seconds has passed since
  # the last inference.
  now = time.time()
  diff = now - last_inference_time
  if diff < interval_between_inference:
    time.sleep(pause_time)
    continue
  last_inference_time = now

  # Load the input audio and run classify.
  tensor_audio.load_from_audio_record(audio_record)
  categories = classifier.classify(tensor_audio)
  os.system('clear') # clear the terminal so we overwrite the output every time
  print("I guess it is:" , end =" ")
  if len(categories) > 1:
    print(categories[0].score*100,"% sure it is",categories[0].label)
  print("Listening again...")



