import torch
import torchaudio
import torch.nn as nn
import numpy as np
import sounddevice as sd
import time

# Configurations
SAMPLE_RATE = 16000  # 16kHz
AUDIO_LENGTH = SAMPLE_RATE * 3  # 3 seconds


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1)  
        )

    def forward(self, x):
        return self.net(x)


model = CNN()
model.load_state_dict(torch.load(
    "../Models/Model_V3.pth",
    map_location=torch.device('cpu')
))
model.eval()


def process_audio(audio_numpy):
    wave = torch.tensor(audio_numpy, dtype=torch.float32).unsqueeze(0)  # dimension [1, N]

    if wave.shape[0] > 1:
        wave = torch.mean(wave, dim=0, keepdim=True)

    if wave.shape[1] < AUDIO_LENGTH:
        size = AUDIO_LENGTH - wave.shape[1]
        wave = torch.nn.functional.pad(wave, (0, size))
    else:
        wave = wave[:, :AUDIO_LENGTH]
    # Create image/spectrogram
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64)
    wave_img = transform(wave)
    wave_img = (wave_img - wave_img.mean()) / wave_img.std()

    return wave_img.unsqueeze(0).to(torch.float32) 



def predict(audio, threshold=0.5):
    input = process_audio(audio)
    with torch.no_grad():
        output = model(input)
        p = torch.sigmoid(output).item()

    return p >= threshold, p


def main():
    print("Listening...\n")
    try:
        while True:
            recording = sd.rec(frames=AUDIO_LENGTH, samplerate=SAMPLE_RATE, channels=1, dtype='float32')# Play microphone
            sd.wait()# Wait for recording

            audio = np.squeeze(recording.T) 

            fartLog, p = predict(audio)# Make prediction

            if fartLog:
                print(f"FART DETECTED!")
            else:
                print(f"UNKNOWN SOUND!")

    except KeyboardInterrupt:
        print("stop")


main()
