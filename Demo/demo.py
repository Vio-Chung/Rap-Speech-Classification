import tensorflow as tf
from tensorflow.keras.models import model_from_json
from keras.layers import LSTM
import torch
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import numpy as np

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


def get_panns_input(audio, sr):

    at = AudioTagging(checkpoint_path=None, device='cuda')
    sed = SoundEventDetection(checkpoint_path=None, device='cuda')
    num_samples = audio.shape[0]
    audio_reshaped = audio[None, :]  # (batch_size, num_samples)
    # Get embeddings for the current audio
    _, embedding = at.inference(audio_reshaped)

    # Get time stamp
    framewise_output = sed.inference(audio_reshaped)

    # repeat the embeddings for every frame
    repeated_embedding = np.repeat(embedding[np.newaxis, :], framewise_output.shape[1], axis=0)
    repeated_embedding_transposed = np.transpose(repeated_embedding, (1, 0, 2))

    return repeated_embedding_transposed



