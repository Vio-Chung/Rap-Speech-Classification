import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import LSTM
import numpy as np
import pandas as pd
import tensorflow as tf
import keras


def load_audio(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)  # sr=None to preserve original sample rate
    return y, sr


def compute_mel_spectrogram(audio, sr, n_mels=128, hop_length=512):
    # Compute Mel spectrogram from the audio signal
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    
    # Convert the Mel spectrogram to dB scale
    logS = librosa.power_to_db(S)

    return logS


def plot_spectrogram(log_spectrogram, sr, hop_length):
    
    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length,
                              x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.show()

def window_audio(audio, sr, audio_seg_size, segments_overlap):
    """
    Segment audio into windows with a specified size and overlap.
    Padding is added only to the last window.

    Parameters
    ----------
    audio : np.ndarray
        The audio signal to be segmented.
    sample_rate : int
        The sampling rate of the audio signal.
    audio_seg_size : float
        The duration of each window in seconds.
    segments_overlap : float
        The duration of the overlap between consecutive windows in seconds.

    Returns
    -------
    audio_windows : list of np.ndarray
        A list of windows of the audio signal.

    Example
    -------
    >>> import librosa
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> audio_windows = window_audio(y, sr, audio_seg_size=1, segments_overlap=0.5)
    """
    # YOUR CODE HERE

    # Calculate the window size in samples
    window_size = int(audio_seg_size * sr)
    # Calculate the overlap size in samples
    overlap_size = int(segments_overlap * sr)
    # Iterate through the audio signal, extracting windows
    audio_windows = []
    start = 0  # Starting index of the window
    while start < len(audio):
        end = start + window_size  # Ending index of the window
        
        # If the window end is within the audio length, extract the window
        if end <= len(audio):
            window = audio[start:end]  # Extract the window

        # Padding the last window with zeros if it extends beyond the audio length
        else:
            window = audio[start:]  # Extract what's left
            window = np.pad(window, (0, window_size - len(window)), 'constant')  # Pad the last window
            
        # Add the window to the list of audio windows
        audio_windows.append(window)
        
        # Update the start position for the next window, considering the overlap
        start += (window_size - overlap_size)
    
    return audio_windows


def wav_generator(df, 
                  # augment,
                  sr,
                  # pitch_shift_steps=2, 
                  shuffle=True):
    """
    Generator function that yields audio and labels from the specified dataset,
    with optional data augmentation.

    Parameters
    ----------
    data_home : str
        The root directory where the dataset is stored.
    augment : bool
        Whether to apply data augmentation (pitch shifting) to the audio.
    track_ids : list of str, optional
        A list of track IDs to load from the dataset, by default None.
        If None, all tracks in the dataset will be loaded.
    sample_rate : int, optional
        The sample rate at which to load the audio, by default 22050.
    pitch_shift_steps : int, optional
        The number of steps by which to shift the pitch for data augmentation, by default 2.
    shuffle : bool, optional
        Whether to shuffle the data before iterating, by default True.

    Yields
    ------
    audio : np.ndarray
        A NumPy array containing the audio waveform data.
    label : int
        The corresponding label for the audio.

    Example
    -------
    >>> data_home = "/path/to/data_directory"
    >>> augment = True
    >>> track_ids = ["track_1", "track_2"]
    >>> generator = wav_generator(data_home, augment, track_ids)
    >>> for audio, label in generator:
    ...     # Process audio and label

    """
    # Hint: base your generator on the win_generator
    # YOUR CODE HERE
    # Get list of audio paths and their corresponding labels
    vocal_paths = df['vocal_path'].tolist()
    labels = df['label'].tolist()

    # Convert labels to numpy array
    
    labels = np.array(labels)

    # Shuffle data
    if shuffle:
        idxs = np.random.permutation(len(labels))
        vocal_paths = [vocal_paths[i] for i in idxs]
        labels = labels[idxs]
        
    for idx in range(len(vocal_paths)):

        # Load audio at given sample_rate and label
        label = labels[idx]
        audio, _ = librosa.load(vocal_paths[idx], sr=sr)

        # Shorten audio to 30sec due to imprecisions in duration (long intro) of rap
        # (ensures same duration files)
        if len(audio) < 60 * sr:
            # Clip or pad audio to make it exactly 30 seconds
            audio = audio[:30*sr]  # Take up to 30 seconds
            if len(audio) < 30 * sr:
                # Pad if shorter than 30 seconds to make it exactly 30 seconds
                audio = np.pad(audio, (0, 30*sr - len(audio)), 'constant')
        else:
            # If longer than 60 seconds, take the segment from 60 to 90 seconds
            audio = audio[60*sr:90*sr]
            # Ensure this segment is exactly 30 seconds
            if len(audio) < 30 * sr:
                audio = np.pad(audio, (0, 30*sr - len(audio)), 'constant')

        # Apply augmentation
        # if augment:
        #     audio = pitch_shift_audio(audio, sample_rate, pitch_shift_steps)
            
        yield audio, label


def win_generator(df, 
                  # augment, 
                  sr, 
                  # pitch_shift_steps=2,
                   n_mels=128, hop_length=512, audio_seg_size=1, segments_overlap=0.5, shuffle=True):
    """
    Generator function that yields Mel spectrograms and labels from the specified dataset,
    with optional data augmentation. 
    Audio is broken down in small windows, the spectrogram is computed and yielded along with the label. 
    The label of the window is assumed to be the same as the label for the entire track.

    Parameters
    ----------
    data_home : str
        The root directory where the dataset is stored.
    augment : bool
        Whether to apply data augmentation (pitch shifting) to the audio.
    track_ids : list of str, optional
        A list of track IDs to load from the dataset, by default None.
        If None, all tracks in the dataset will be loaded.
    sample_rate : int, optional
        The sample rate at which to load the audio, by default 22050.
    pitch_shift_steps : int, optional
        The number of steps by which to shift the pitch for data augmentation, by default 2.
    n_mels : int, optional
        The number of Mel bands to generate, by default 128.
    hop_length : int, optional
        The number of samples between successive frames, by default 512.
    audio_seg_size : float, optional
        The size of audio segments in seconds, by default 1.
    segments_overlap : float, optional
        The overlap between audio segments in seconds, by default 0.5.
    shuffle : bool, optional
        Whether to shuffle the data before iterating, by default True.

    Yields
    ------
    spectrogram : np.ndarray
        A NumPy array containing the Mel spectrogram data.
    label : int
        The corresponding label for the spectrogram.

    Example
    -------
    >>> data_home = "/path/to/data_directory"
    >>> augment = True
    >>> track_ids = ["track_1", "track_2"]
    >>> generator = win_generator(data_home, augment, track_ids)
    >>> for spectrogram, label in generator:
    ...     # Process spectrogram and label
    """

    # Get list of audio paths and their corresponding labels
    vocal_paths = df['vocal_path'].tolist()
    labels = df['label'].tolist()

    # Convert labels to numpy array
    
    labels = np.array(labels)

    # Shuffle data
    if shuffle:
        idxs = np.random.permutation(len(labels))
        vocal_paths = [vocal_paths[i] for i in idxs]
        labels = labels[idxs]


    for idx in range(len(vocal_paths)):

        # Load audio at given sample_rate and label
        label = labels[idx]
        audio, _ = librosa.load(vocal_paths[idx], sr=sr)

        # Shorten audio to 29s due to imprecisions in duration of GTZAN
        # (ensures same duration files)
        audio = audio[60*sr:90*sr]

        # Apply augmentation
        # if augment:
        #     audio = pitch_shift_audio(audio, sample_rate, pitch_shift_steps)

        # Compute audio windowing
        audio_windows = window_audio(audio, sr, audio_seg_size, segments_overlap)

        # Loop over windows
        for window in audio_windows:
            
            # Compute Mel spectrogram
            spectrogram = compute_mel_spectrogram(window, sr, n_mels, hop_length)
            spectrogram = np.expand_dims(spectrogram, axis=2)

            yield spectrogram, label



def create_dataset(data_generator, input_args):
    """
    Create a dataset from a generator function that yields spectrograms and labels.

    Parameters
    ----------
    data_generator : function
        A generator function that yields spectrograms and labels.
    input_args : list
        A list of arguments to pass to the data_generator function.

    Returns
    -------
    X : np.ndarray
        A NumPy array containing all spectrograms.
    Y : np.ndarray
        A NumPy array containing all labels.
    """
    spectrograms = []
    labels = []

    # Call the generator with the provided arguments
    for spectrogram, label in data_generator(*input_args):
        spectrograms.append(spectrogram)
        labels.append(label)

    # Convert lists to NumPy arrays
    X = np.array(spectrograms)
    Y = np.array(labels)

    return X, Y

def reload_lstm(model_path, weights_path, optimizer, loss, metrics):

    # Load the best checkpoint of the model from json file (due to custom layers)
    loaded_json = open(model_path, 'r').read()
    reloaded_model = model_from_json(loaded_json, custom_objects={'LSTM': LSTM})

    reloaded_model.compile(optimizer=optimizer,
                            loss=loss, 
                            metrics=metrics)
    # restore weights
    reloaded_model.load_weights(weights_path)

    return reloaded_model

def plot_loss(history):
    
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()