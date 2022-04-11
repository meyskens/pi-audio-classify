# TensorFlow Lite Python audio classification example with Raspberry Pi.

This example uses [TensorFlow Lite](https://tensorflow.org/lite) with Python on
a Raspberry Pi to perform real-time audio classification using audio streamed
from an USB microphone.

## Installation

We need Tensorflow Lite installed.
To install this on your Raspberry Pi, follow the instructions in the
[Python quickstart](https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python).

Next we need to clone this project on the Raspberry Pi:
```bash
git clone https://github.com/meyskens/pi-audio-classify.git
cd pi-audio-classify
```

Install the dependencies for this project:

```bash
pip install -r requirements.txt
sudo apt-get install libatlas-base-dev
```

## Prepare the model

This script is made to use the [Google Teachable Machine for Audio Classification](https://teachablemachine.withgoogle.com/train/audio). Follow the instructions there and download it as "Tensorflow Lite". 

*TIP: use the same USB microphone on your laptop for best results!*

You will get a zip file. Unzip it and copy the `soundclassifier_with_metadata.tflite` file to the directory you cloned on the Raspberry Pi.

## Usage

You can now run the script and get output!

```bash
$ python classify.py
```


## Credits

Big thanks to the Tensorflow authors for their super clear examples on which this project is based! 