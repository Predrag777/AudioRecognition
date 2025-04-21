import torch
import torchaudio
import torch.nn as nn

# Configuration
SAMPLE_RATE = 16000 # 16kHz
AUDIO_LENGTH = SAMPLE_RATE * 5  # 5 seconds

# Initialize Convolutional neural network with three convolutional layers
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,stride=1,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(),
            nn.Linear(64,12)
        )

    def forward(self,x):
        return self.net(x)




def preprocess_audio(file_path):
    audio_wave,sample_rate=torchaudio.load(file_path, normalize=True) # load audio

    if sample_rate != SAMPLE_RATE:# if sample is not deserved rate, resample it
        resampler=torchaudio.transforms.Resample(sample_rate,SAMPLE_RATE)
        audio_wave=resampler(audio_wave)# new audio wave

    mono_audio=torch.mean(audio_wave, dim=0, keepdim=True) # from stereo, create monophonic audio => from two channels (stereo) to one channel (monophonic)

    if mono_audio.shape[1]<AUDIO_LENGTH: # duration is less than deserved
        size=AUDIO_LENGTH-mono_audio.shape[1] # size need to be added to audio
        mono_audio=torch.nn.functional.pad(mono_audio,(0,size)) # add zeros to the audio to match the desired length
    else:
        mono_audio=mono_audio[:,:AUDIO_LENGTH] # If length is bigger, just crop to AUDIO_LENGTH size

    transform=torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64) # Create transformation for mel spectrogram
    spectrogram=transform(mono_audio) # Spectrogram
    spectrogram=(spectrogram-spectrogram.mean())/spectrogram.std() # Normalize it for better learning

    return spectrogram.unsqueeze(0).to(torch.float32)  # [1, 1, 64, Time]



model=CNN()
model.load_state_dict(torch.load("../Models/Model_V2.pth")) # Load weights
##########################################################
for i in range(1, 13):
    file_path=f"../Test/simpleFartAudio{i}.mp3"
    input=preprocess_audio(file_path)
    with torch.no_grad():
        output=model(input)
        prediction=torch.argmax(output,dim=1).item()

    if prediction==0: # Find deserved class
        print("FART!")
    else:
        print("UNKNOWN!")
