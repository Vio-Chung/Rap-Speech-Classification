from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     Flatten, Dense, Layer, BatchNormalization,
                                     Dropout, Reshape, TimeDistributed, LSTM)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import model_from_json
from tensorflow import keras
import keras.backend as K
import tensorflow as tf
from tcn import TCN
import numpy as np
import tensorflow_hub as hub
import openl3
import IPython
import soundfile as SF




def cnn_model(input_shape):
    """
    Create a Convolutional Neural Network (CNN) model for time series classification.

    The CNN architecture consists of three convolutional blocks followed by fully connected layers.
    Each convolutional block has the following layers:
    1. Conv2D layer with 20 filters, a kernel size of (3, 3), and ReLU activation
    2. MaxPooling2D layer with a pool size of (3, 1)
    3. Dropout layer with a dropout rate of 0.15

    After the convolutional blocks, the following fully connected layers are added:
    1. Flatten layer to convert the 2D feature maps into a 1D feature vector
    2. Dense layer with 64 units and ReLU activation
    3. Dense layer with 32 units and ReLU activation
    4. Dense layer with 10 units and softmax activation (output layer)

    The model is compiled using the Adam optimizer, sparse categorical crossentropy loss,
    and accuracy metric.

    Parameters
    ----------
    input_shape : tuple
      The shape of the input data, including the number of time steps and the number of features.
      For example, (100, 10) means 100 time steps and 10 features.

    Returns
    -------
    model : keras.Sequential
      A compiled Keras Sequential model with the CNN architecture.

    Example
    -------
    >>> input_shape = (100, 10)  # 100 time steps and 10 features
    >>> model = cnn_model(input_shape)
    >>> print(model.summary())

    # Prepare your dataset
    >>> x_train, y_train, x_test, y_test = ...  # Load or preprocess your data

    # Train the model
    >>> history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                            validation_data=(x_test, y_test))

    # Evaluate the model
    >>> loss, accuracy = model.evaluate(x_test, y_test)
    >>> print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # Make predictions
    >>> y_pred = model.predict(x_test)
    >>> y_pred_classes = np.argmax(y_pred, axis=1)
    """
    # Note: Please use the Sequential API (i.e., keras.Sequential)
    # when building your model; do not use the Functional API.
    # YOUR CODE HERE
    from tensorflow.keras.models import Sequential
    model = Sequential()
    
    # First convolutional layer
        
    model.add(Conv2D(20, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 1)))
    model.add(Dropout(0.15))
    
    # Second convolutional layer
    model.add(Conv2D(20, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 1)))
    model.add(Dropout(0.15))
    
    # Third convolutional layer
    model.add(Conv2D(20, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 1)))
    model.add(Dropout(0.15))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    

    model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
    return model


def lstm_model(input_shape, learning_rate=None):
    model = Sequential([
        LSTM(units=128, input_shape=input_shape),
        Dropout(0.2),
        Dense(units=64, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])

    if learning_rate is None:
        opt = keras.optimizers.Adam()
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
