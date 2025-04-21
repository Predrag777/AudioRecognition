# Audio recognizer

## How to run
Run python script **main.py** from Code folder. **Test** folder contains test cases. Folders **DogBarking** and **CarHornVoice** contain a false test cases. 

## Idea
The input is an audio clip of 5 seconds. If the audio is shorter than 5 seconds, we extend it to 5 seconds. If the audio is longer than 5 seconds, we trim it down to 5 seconds. Each second is represented by 16,000 samples (16kHz), which is a usually used sampling rate in audio processing. The audio is then represented using the Mel scale. The Mel scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another. The reference point between this scale and the normal frequency measurement is defined by equating a 1000 Hz tone, 40 dB above the listener's threshold, with a pitch of 1000 mels. Using Short-Time Fourier Transform (STFT), we create a spectrogram, and then apply a Mel filter bank to convert it into a Mel spectrogram.


Further, we need to analyze given spectogram. For image processing, we usually use **Convolutional Neural Network**. We have our targeted audio of the fart for recognition. For that purposes, we created 11 more classes of different audio, to provide our model more samples to know what is fart, and what is not. At the end, we have only one targetted class (of the fart).


## Code explaination
Lines 8 and 9 are for configuration of sample rate and audio length. Sample rate is 16000 (16kHz) and audio length is 5 seconds. From line 11, we define CNN model structure. Our CNN contains 3 convolutional layers with ReLU activation functions. Function **process_audio** create deserved audio shape and derive spectrogram from it. If inputed audio is not 16kHz we make it to be 16kHz. Next, we convert audio from stereo (two channels) to be monophonic (one channel), because we are not interested in directions of the sound. Only on its content. If audio is not 5 seconds length, we create it to be. Last one is transforming sound in mel spectrograma and normalize it. 


On line 60, the pre-trained model's weights are loaded, so the CNN can make predictions without re-training. Class 0 is targeted class. We are not interested for other classes.
