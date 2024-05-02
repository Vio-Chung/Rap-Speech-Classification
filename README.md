# Rap-Speech-Classification

Numerous studies have explored music and speech classification by using a range of different techniques from digital signal processing to machine learning. These models often capitalize on the harmonic patterns prevalent in music to distinguish it from speech, generally yielding high accuracy. 
However, a genre like rap, which shares similarities with spoken words in its vocals, blurs these lines with its speech-like qualities. This project will investigate the efficacy of 4 existing pre-trained models + LSTM and 1 CNN+FC (fully connected layers) in discriminating between rap vocals and speech. 
Our data is self-collected audio data of speech and rap vocals, which are scrapped from youtube via `yt-dlp`, followed by `Demucs` for separating target vocals from music tracks.
