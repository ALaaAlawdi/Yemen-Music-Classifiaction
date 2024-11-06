import numpy as np
import logging
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
# Import necessary libraries
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import numpy as np
import logging
import pandas as pd 
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model
import numpy as np
import logging
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Add, GlobalAveragePooling1D, Activation, ReLU, SeparableConv1D
import pandas as pd
import os
import pickle
import tensorflow as tf 
from tensorflow.keras import layers 
import argparse



# # Function to load trained model weights
# def load_trained_model_weights(model, model_name, experiment_folder):
#     """
#     Load the weights of a trained model from a specified folder.
    
#     Args:
#     - model: The model architecture into which the weights will be loaded.
#     - model_name: The name of the model (used to locate the weights file).
#     - experiment_folder: The folder where the model's weights are saved.
    
#     Returns:
#     - model: The model with loaded weights.
#     """
#     # Path to the saved weights
#     weights_path = os.path.join(experiment_folder, f"{model_name}_weights.weights.h5")
    
#     # Load the weights into the model
#     model.load_weights(weights_path)
#     logging.info(f"Weights for {model_name} loaded from {weights_path}")
    
#     return model

# # Function to load the features and make predictions
# def make_predictions(features_file, model, label_encoder):
#     # Load features from the .npy file
#     features = np.load(features_file)

#     # Reshape features for prediction
#     input_feature = features.reshape(1, features.shape[0], 1)

#     # Make the prediction
#     predicted_label = np.argmax(model.predict(input_feature), axis=-1)

#     # Decode the predicted label to its original form
#     decoded_label = label_encoder.inverse_transform(predicted_label)

#     return decoded_label[0]
#########################################################################################
#                                                VGG16                                  #
#########################################################################################
def VGG16_1D(input_shape):
    model = Sequential()

    # First block
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))  # Added padding='same' to avoid shrinking

    # Second block
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))  # Added padding='same'

    # Third block
    model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))  # Added padding='same'

    # Flatten before Dense layers
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer (adjust for the number of classes)
    model.add(Dense(units=5, activation='softmax'))

    return model


############################################################################################
#                                                AlexNet                                  #
############################################################################################
def AlexNet_1D(input_shape):
    model = Sequential()

    # First block
    model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))  # Add padding='same' to avoid shrinking

    # Second block
    model.add(Conv1D(filters=192, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))  # Add padding='same'

    # Third block

    model.add(Conv1D(filters=384, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))  # Add padding='same'

    # Flatten before Dense layers
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(5 , activation='softmax'))

    return model

from tensorflow.keras.layers import Input
#########################################################################################
#                                                VGG19                                  #
##########################################################################################
def VGG19_1D(input_shape):
    input_layer = Input(shape=input_shape)

    # Block 1
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # Block 2
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # Block 3
    x = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # Block 4
    x = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # Block 5
    x = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    # Block 6 - Fully connected layers
    x = Flatten()(x)
    x = Dense(units=2048, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=2048, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=5 , activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x, name='VGG19_1D')
    return model
####################################################
#                  ResNet50_1D
####################################################
# Define 1D Convolutional Block
def conv_block_1d(x, filters, kernel_size, strides, padding='same'):
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# Identity Block for ResNet
def identity_block_1d(x, filters):
    shortcut = x
    x = conv_block_1d(x, filters=filters, kernel_size=1, strides=1)
    x = conv_block_1d(x, filters=filters, kernel_size=3, strides=1)
    x = Conv1D(filters=filters * 4, kernel_size=1)(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x
# Projection Block for ResNet
def projection_block_1d(x, filters, strides):
    shortcut = x
    x = conv_block_1d(x, filters=filters, kernel_size=1, strides=strides)
    x = conv_block_1d(x, filters=filters, kernel_size=3, strides=1)
    x = Conv1D(filters=filters * 4, kernel_size=1)(x)
    x = BatchNormalization()(x)
    shortcut = Conv1D(filters=filters * 4, kernel_size=1, strides=strides)(shortcut)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

# Define the ResNet50 model architecture
def ResNet50_1D(input_shape):
    inputs = Input(shape=input_shape)

    # Initial conv layer
    x = Conv1D(filters=64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Conv block 1
    x = projection_block_1d(x, filters=64, strides=1)
    x = identity_block_1d(x, filters=64)
    x = identity_block_1d(x, filters=64)

    # Conv block 2
    x = projection_block_1d(x, filters=128, strides=2)
    x = identity_block_1d(x, filters=128)
    x = identity_block_1d(x, filters=128)
    x = identity_block_1d(x, filters=128)

    # Conv block 3
    x = projection_block_1d(x, filters=256, strides=2)
    x = identity_block_1d(x, filters=256)
    x = identity_block_1d(x, filters=256)
    x = identity_block_1d(x, filters=256)
    x = identity_block_1d(x, filters=256)
    x = identity_block_1d(x, filters=256)

    # Conv block 4
    x = projection_block_1d(x, filters=512, strides=2)
    x = identity_block_1d(x, filters=512)
    x = identity_block_1d(x, filters=512)

    # Global average pooling and dense layer
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(5, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model



#################################################################
#                          MobileNet
#################################################################
# Define the MobileNet 1D model
def mobilenet_1d(input_shape):
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Convolutional layers
    x = layers.Conv1D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise separable convolutions
    x = layers.SeparableConv1D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv1D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv1D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv1D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv1D(512, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(5):
        x = layers.SeparableConv1D(512, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.SeparableConv1D(1024, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Output layer
    outputs = layers.Dense( 5 , activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
############################################################################################
############################################################################################


# if __name__ == "__main__":
#     input_shape = (40, 1)  # Adjust based on your feature size
#     model_list = ['VGG19', 'VGG16', 'AlexNet', 'ResNet50', 'MobileNet']

#     for model_name in model_list:
#         print(f"Processing {model_name} model...")

#         # Initialize the correct model based on the model_name
#         if model_name == 'VGG19':
#             model = VGG19_1D(input_shape=input_shape)
#         elif model_name == 'VGG16':
#             model = VGG16_1D(input_shape=input_shape)
#         elif model_name == 'AlexNet':
#             model = AlexNet_1D(input_shape=input_shape)
#         elif model_name == 'ResNet50':
#             model = ResNet50_1D(input_shape=input_shape)
#         elif model_name == 'MobileNet':
#             model = mobilenet_1d(input_shape=input_shape)

#         # Load the saved weights
#         experiment_folder = r'C:\Users\bravo\Documents\yemen music\results'
#         model = load_trained_model_weights(model, model_name, experiment_folder)

#         # Load the label encoder
#         with open(r'C:\Users\bravo\Documents\yemen music\label_encoder.pkl', 'rb') as f:
#             label_encoder = pickle.load(f)

#         # Ensure the label encoder contains class labels
#         if not hasattr(label_encoder, 'classes_'):
#             raise ValueError("The loaded label encoder does not contain 'classes_' attribute.")

#         # Path to the features file extracted earlier
#         features_file = './snany_processed_features.npy'

#         # Make the prediction
#         predicted_label = make_predictions(features_file, model, label_encoder)

#         print(f"Predicted label for {model_name}: {predicted_label}")



# Function to load trained model weights
def load_trained_model_weights(model, model_name, experiment_folder):
    """
    Load the weights of a trained model from a specified folder.
    """
    weights_path = os.path.join(experiment_folder, f"{model_name}_weights.weights.h5")
    model.load_weights(weights_path)
    logging.info(f"Weights for {model_name} loaded from {weights_path}")
    return model

# Function to load the features and make predictions
def make_predictions(features_file, model, label_encoder):
    """
    Load features from a .npy file and make predictions using the trained model.
    """
    features = np.load(features_file)
    input_feature = features.reshape(1, features.shape[0], 1)
    predicted_label = np.argmax(model.predict(input_feature), axis=-1)
    decoded_label = label_encoder.inverse_transform(predicted_label)
    return decoded_label[0]

# Parser for command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Make predictions using pre-trained models")
    parser.add_argument('--features_file', type=str, required=True, help="Path to the features .npy file")
    parser.add_argument('--experiment_folder', type=str, required=True, help="Path to the folder with the trained model weights")
    parser.add_argument('--label_encoder', type=str, required=True, help="Path to the label encoder .pkl file")
    return parser.parse_args()

# # Function to initialize all models and run predictions
# def run_all_models(input_shape, features_file, experiment_folder, label_encoder):
#     models = {
#         'VGG19': VGG19_1D(input_shape=input_shape),
#         'VGG16': VGG16_1D(input_shape=input_shape),
#         'AlexNet': AlexNet_1D(input_shape=input_shape),
#         'ResNet50': ResNet50_1D(input_shape=input_shape),
#         'MobileNet': mobilenet_1d(input_shape=input_shape)
#     }

#     for model_name, model in models.items():
#         print(f"Processing {model_name} model...")

#         # Load the saved weights
#         model = load_trained_model_weights(model, model_name, experiment_folder)

#         # Make the prediction
#         predicted_label = make_predictions(features_file, model, label_encoder)

#         print(f"Predicted label for {model_name}: {predicted_label}")
def run_all_models(input_shape, features_file, experiment_folder, label_encoder):
    models = {
        'VGG19': VGG19_1D(input_shape=input_shape),
        'VGG16': VGG16_1D(input_shape=input_shape),
        'AlexNet': AlexNet_1D(input_shape=input_shape),
        'ResNet50': ResNet50_1D(input_shape=input_shape),
        'MobileNet': mobilenet_1d(input_shape=input_shape)
    }

    predicted_labels = {}

    for model_name, model in models.items():
        print(f"Processing {model_name} model...")

        # Load the saved weights
        model = load_trained_model_weights(model, model_name, experiment_folder)

        # Make the prediction
        predicted_label = make_predictions(features_file, model, label_encoder)
        predicted_labels[model_name] = predicted_label

        print(f"Predicted label for {model_name}: {predicted_label}")  # Print for debugging

    return predicted_labels  # Ensure it returns the dictionary


if __name__ == "__main__":
    args = parse_args()

    input_shape = (40, 1)  # Adjust based on your feature size

    # Load the label encoder
    with open(args.label_encoder, 'rb') as f:
        label_encoder = pickle.load(f)

    # Ensure the label encoder contains class labels
    if not hasattr(label_encoder, 'classes_'):
        raise ValueError("The loaded label encoder does not contain 'classes_' attribute.")

    # Run predictions for all models
    run_all_models(input_shape, args.features_file, args.experiment_folder, label_encoder)


# python prediction.py --features_file ./snany_processed_features.npy --experiment_folder ./results --label_encoder ./label_encoder.pkl
