import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from tensorflow.python.keras import Sequential

class NeuralNetwork(object):

    def __init__(self):
        self.model = None

    def createModel(self):
        """Create and compile the keras model. See layers-18pct.cfg and 
           layers-params-18pct.cfg for the network model, 
           and https://code.google.com/archive/p/cuda-convnet/wikis/LayerParams.wiki 
           for documentation on the layer format.
        """
        self.model = Sequential()
        self.model.add(keras.layers.Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same',
                                           input_shape=(32, 32, 3), data_format="channels_last", dilation_rate=(1, 1),
                                           activation=tf.nn.relu))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        self.model.add(keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, ))
        self.model.add(keras.layers.Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same',
                                           dilation_rate=(1, 1), activation=tf.nn.relu))
        self.model.add(keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        self.model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
        self.model.add(keras.layers.Conv2D(filters=32, kernel_size=5, strides=(1, 1), padding='same',
                                           dilation_rate=(1, 1), activation=tf.nn.relu))
        self.model.add(keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    def train(self, train_data, train_labels, epochs):
        """Train the keras model
        
        Arguments:
            train_data {np.array} -- The training image data
            train_labels {np.array} -- The training labels
            epochs {int} -- The number of epochs to train for
        """

        self.model.fit(train_data, train_labels, epochs=epochs, batch_size=128)

        pass

    def evaluate(self, eval_data, eval_labels):
        """Calculate the accuracy of the model
        
        Arguments:
            eval_data {np.array} -- The evaluation images
            eval_labels {np.array} -- The labels for the evaluation images
        """

        return self.model.evaluate(eval_data, eval_labels)[1]

        pass

    def test(self, test_data):
        """Make predictions for a list of images and display the results
        
        Arguments:
            test_data {np.array} -- The test images
        """

        return self.model.predict(test_data)

        pass

    ## Exercise 7 Save and load a model using the keras.models API
    def saveModel(self, saveFile="model.h5"):
        """Save a model using the keras.models API
        
        Keyword Arguments:
            saveFile {str} -- The name of the model file (default: {"model.h5"})
        """

        keras.models.save_model(self.model, saveFile)

        pass

    def loadModel(self, saveFile="model.h5"):
        """Load a model using the keras.models API
        
        Keyword Arguments:
            saveFile {str} -- The name of the model file (default: {"model.h5"})
        """

        self.model = keras.models.load_model(saveFile)

        pass
