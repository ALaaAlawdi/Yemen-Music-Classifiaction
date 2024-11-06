import os
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter
import xgboost as xgb
from  tensorflow.keras import Model
import tensorflow as tf 
from tensorflow.keras import layers 
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling1D, Add, SeparableConv1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to ensure the output directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to save classification report
def save_classification_report(report, folder, filename):
    ensure_dir(folder)
    with open(os.path.join(folder, filename), 'w') as f:
        f.write(report)

# Function to save plots
def save_plot(fig, folder, filename, dpi=300):
    ensure_dir(folder)
    fig.savefig(os.path.join(folder, filename), dpi=dpi)

# Function to plot label distribution
def plot_label_distribution(y, label_names, folder, stage="before"):
    label_counts = np.bincount(y)
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("Set2", len(label_names))
    bars = sns.barplot(x=label_names, y=label_counts, ax=ax, palette=colors, edgecolor='black')

    for bar, count in zip(bars.patches, label_counts):
        ax.annotate(f'{count}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                    ha='center', va='center', size=12, xytext=(0, 8), textcoords='offset points', weight='bold')

    ax.set_title(f"Label Distribution {stage.capitalize()} Undersampling", fontsize=18, weight='bold')
    ax.set_ylabel("Count", fontsize=14)
    ax.set_xlabel("Labels", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    save_plot(fig, folder, f'label_distribution_{stage}.png', dpi=300)

# Function to undersample data for balancing
def undersample_data(X, y):
    counter = Counter(y)
    min_count = min(counter.values())
    indices_to_keep = []
    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        selected_indices = np.random.choice(label_indices, min_count, replace=False)
        indices_to_keep.extend(selected_indices)
    X_balanced = X[indices_to_keep]
    y_balanced = y[indices_to_keep]
    return X_balanced, y_balanced

# Model training and evaluation function
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, folder, label_names):
    logging.info(f"Training {model_name} model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_names)
    conf_matrix = confusion_matrix(y_test, y_pred)

    save_classification_report(report, folder, f'{model_name}_classification_report.txt')

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=label_names, yticklabels=label_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{model_name} Confusion Matrix')
    save_plot(fig, folder, f'{model_name}_confusion_matrix.png')

    accuracy = model.score(X_test, y_test)
    logging.info(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Accuracy: {accuracy:.4f}")

# Deep learning model training function
def train_and_evaluate_dl_model(model, model_name, X_train, X_test, y_train, y_test, folder, label_names, epochs=250, patience=15, batch_size=16):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stopping], batch_size=batch_size)
    
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred, target_names=label_names)
    conf_matrix = confusion_matrix(y_test, y_pred)

    save_classification_report(report, folder, f'{model_name}_classification_report.txt')

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=label_names, yticklabels=label_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{model_name} Confusion Matrix')
    save_plot(fig, folder, f'{model_name}_confusion_matrix.png')

    # Training loss and accuracy plots
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_name} Loss')
    ax.legend()
    save_plot(fig, folder, f'{model_name}_loss_plot.png')

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(history.history['accuracy'], label='Training Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name} Accuracy')
    ax.legend()
    save_plot(fig, folder, f'{model_name}_accuracy_plot.png')

    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    logging.info(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Accuracy: {accuracy:.4f}")

    # Save the model weights
    model.save_weights(os.path.join(folder, f'{model_name}_weights.weights.h5'))


# Main function to process arguments and run the training
def main(args):
    # Load features and labels
    features = np.load(args.features)
    labels = np.load(args.labels)

    # Process labels (remove '_10s')
    new_labels = [label.replace('_10s', '') for label in labels]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(new_labels)
    label_names = label_encoder.inverse_transform(np.unique(y_encoded))

    # Reshape X to 2D (if 3D), then split into training and test sets
    X_combined = features
    if X_combined.ndim == 3:
        n_samples, n_time_steps, n_features = X_combined.shape
        X_combined = X_combined.reshape(n_samples, n_time_steps * n_features)

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=args.test_size, random_state=42)

    # Experiment folder
    ensure_dir(args.output_dir)

    # Plot label distribution before undersampling
    plot_label_distribution(y_train, label_names, args.output_dir, stage="before")

    # Optionally perform undersampling
    if args.undersample:
        X_train, y_train = undersample_data(X_train, y_train)

    # Plot label distribution after undersampling
    plot_label_distribution(y_train, label_names, args.output_dir, stage="after")

    # Train traditional XGBoost model
    xgb_model = xgb.XGBClassifier(n_estimators=args.n_estimators, random_state=42)
    train_and_evaluate_model(xgb_model, 'XGBoost', X_train, X_test, y_train, y_test, args.output_dir, label_names)

    # Reshape for deep learning models (1D CNN)
    X_train_dl = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_dl = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Train VGG19 model
    VGG19_model = VGG19_1D(input_shape=(X_train_dl.shape[1], 1))
    VGG19_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_and_evaluate_dl_model(VGG19_model, 'VGG19', X_train_dl, X_test_dl, y_train, y_test, args.output_dir, label_names, epochs=args.epochs)

    #VGG16 Model for 1D data
    vgg16_1d_model = VGG16_1D(input_shape=(X_train_dl.shape[1], 1))
    vgg16_1d_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_and_evaluate_dl_model(vgg16_1d_model, 'VGG16', X_train_dl, X_test_dl, y_train, y_test, args.output_dir, label_names, epochs=args.epochs)

    # MobileNet Model for 1D data
    mobilenet_1d_model = mobilenet_1d(input_shape=(X_train_dl.shape[1], 1))
    mobilenet_1d_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_and_evaluate_dl_model(mobilenet_1d_model, 'Mobilenet', X_train_dl, X_test_dl, y_train, y_test, args.output_dir, label_names, epochs=args.epochs)

    # ResNet50 Model for 1D data
    ResNet50_1D_model = ResNet50_1D(input_shape=(X_train_dl.shape[1], 1))
    ResNet50_1D_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_and_evaluate_dl_model(ResNet50_1D_model, 'ResNet50', X_train_dl, X_test_dl, y_train, y_test, args.output_dir, label_names, epochs=args.epochs)

    # AlexNet Model for 1D data
    alexnet_1d_model = AlexNet_1D(input_shape=(X_train_dl.shape[1], 1))
    alexnet_1d_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    train_and_evaluate_dl_model(alexnet_1d_model, 'AlexNet', X_train_dl, X_test_dl, y_train, y_test, args.output_dir, label_names, epochs=args.epochs)

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
    model.add(Dense(units= 5 , activation='softmax'))

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
    model.add(Dense( 5 , activation='softmax'))

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


############################################################################################
#                                                Resent50                                  #
############################################################################################
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

# Define ResNet50 1D
def ResNet50_1D(input_shape):
    inputs = Input(shape=input_shape)

    # Initial conv layer
    x = conv_block_1d(inputs, filters=64, kernel_size=7, strides=2, padding='same')
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
    outputs = Dense( 5  , activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


#####################################################################################
#                          MobileNet
######################################################################################
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


# # Define helper functions
# def save_classification_report(report, folder, filename):
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     with open(os.path.join(folder, filename), 'w') as f:
#         f.write(report)

# def save_plot(fig, folder, filename):
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     fig.savefig(os.path.join(folder, filename), dpi=300)  

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import logging

# def plot_label_distribution(y, label_names, experiment_folder, stage="before"):
#     label_counts = np.bincount(y)

#     # Create the plot with a professional style
#     sns.set(style="whitegrid")
#     fig, ax = plt.subplots(figsize=(12, 8))  # Large figure size for better readability

#     # Create a custom color palette
#     colors = sns.color_palette("Set2", len(label_names))

#     # Create the bar plot with enhanced style
#     bars = sns.barplot(x=label_names, y=label_counts, ax=ax, palette=colors, edgecolor='black')

#     # Add annotations to the bars (number of samples)
#     for bar, count in zip(bars.patches, label_counts):
#         ax.annotate(f'{count}',
#                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
#                     ha='center', va='center', size=12, xytext=(0, 8),
#                     textcoords='offset points', color='black', weight='bold')

#     # Customize the grid, labels, and title for a professional look
#     ax.grid(True, which='major', linestyle='--', linewidth=0.75)
#     ax.set_axisbelow(True)
#     ax.set_title(f"Label Distribution {stage.capitalize()} Undersampling", fontsize=18, weight='bold')
#     ax.set_ylabel("Count", fontsize=14)
#     ax.set_xlabel("Labels", fontsize=14)

#     # Rotate the x-axis labels for better readability (especially for longer labels)
#     plt.xticks(rotation=45, ha='right', fontsize=12)
#     plt.yticks(fontsize=12)

#     # Add a tight layout to make sure everything fits well
#     plt.tight_layout()

#     # Log the number of samples for each label
#     for label, count in zip(label_names, label_counts):
#         logging.info(f"{label}: {count} samples")

#     # Save the plot in high resolution (300 DPI)
#     save_plot(fig, experiment_folder, f'label_distribution_{stage}.png', dpi=300)

# def undersample_data(X, y):
#     counter = Counter(y)
#     min_count = min(counter.values())
#     logging.info(f"Minimum samples across all labels: {min_count}")

#     indices_to_keep = []
#     for label in np.unique(y):
#         label_indices = np.where(y == label)[0]
#         selected_indices = np.random.choice(label_indices, min_count, replace=False)
#         indices_to_keep.extend(selected_indices)

#     X_balanced = X[indices_to_keep]
#     y_balanced = y[indices_to_keep]
#     return X_balanced, y_balanced

# def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, experiment_folder, label_names):
#     logging.info(f"Training {model_name} model...")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     report = classification_report(y_test, y_pred, target_names=label_names)
#     conf_matrix = confusion_matrix(y_test, y_pred)

#     # Save classification report
#     save_classification_report(report, experiment_folder, f'{model_name}_classification_report.txt')

#     # Save confusion matrix plot
#     fig, ax = plt.subplots(figsize=(10, 8))  # Increase the size of the confusion matrix plot
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
#                 xticklabels=label_names, yticklabels=label_names)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('True')
#     ax.set_title(f'{model_name} Confusion Matrix')
#     save_plot(fig, experiment_folder, f'{model_name}_confusion_matrix.png')

#     accuracy = model.score(X_test, y_test)
#     logging.info(f"{model_name} Accuracy: {accuracy:.4f}")
#     print(f"{model_name} Accuracy: {accuracy:.4f}")



# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix

# def save_plot(fig, folder, filename, dpi=300):
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     # Save the figure with high resolution (default: 300 DPI)
#     fig.savefig(os.path.join(folder, filename), dpi=dpi)


# import os
# import logging
# from tensorflow.keras.models import save_model as keras_save_model  # Import for saving full Keras model

# def save_model(model, model_name, folder):
#     """
#     Save the full model.

#     Arguments:
#     - model: The trained model object (TensorFlow/Keras in this case).
#     - model_name: Name of the model (for filename).
#     - folder: Directory where the model should be saved.
#     """
#     if not os.path.exists(folder):
#         os.makedirs(folder)

#     model_path = os.path.join(folder, model_name)

#     # Save the entire model (architecture + weights) as an H5 file
# #     keras_save_model(model, model_path + '.h5')
#     logging.info(f"Keras model saved at {model_path}.h5")


# def train_and_evaluate_dl_model(model, model_name, X_train, X_test, y_train, y_test, experiment_folder, label_names, epochs=250, patience=15, batch_size=16):
#     early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

#     history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test)  , callbacks=[early_stopping] , batch_size=batch_size)
#     y_pred = np.argmax(model.predict(X_test), axis=1)
#     report = classification_report(y_test, y_pred, target_names=label_names)
#     conf_matrix = confusion_matrix(y_test, y_pred)

#     # Save classification report
#     save_classification_report(report, experiment_folder, f'{model_name}_classification_report.txt')

#     # Save confusion matrix plot
#     fig, ax = plt.subplots(figsize=(10, 8))  # Increase the size of the confusion matrix plot
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
#                 xticklabels=label_names, yticklabels=label_names)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('True')
#     ax.set_title(f'{model_name} Confusion Matrix')
#     save_plot(fig, experiment_folder, f'{model_name}_confusion_matrix.png', dpi=300)  # High resolution

#     # Save separate loss plot
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.plot(history.history['loss'], label='Training Loss')
#     ax.plot(history.history['val_loss'], label='Validation Loss')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Loss')
#     ax.set_title(f'{model_name} Loss')
#     ax.legend()
#     save_plot(fig, experiment_folder, f'{model_name}_loss_plot.png', dpi=300)  # High resolution

#     # Save separate accuracy plot
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.plot(history.history['accuracy'], label='Training Accuracy')
#     ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Accuracy')
#     ax.set_title(f'{model_name} Accuracy')
#     ax.legend()
#     save_plot(fig, experiment_folder, f'{model_name}_accuracy_plot.png', dpi=300)  # High resolution

#     accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
#     logging.info(f"{model_name} Accuracy: {accuracy:.4f}")
#     print(f"{model_name} Accuracy: {accuracy:.4f}")

#     # Save the model's weights for later use
#     model.save_weights(f'{experiment_folder}/{model_name}_weights.h5')
#     logging.info(f"Weights for {model_name} saved at {experiment_folder}/{model_name}_weights.h5")

#     # Save the full model (architecture + weights)
#     save_model(model, model_name, experiment_folder)


# import numpy as np

# features = np.load(r'C:\Users\bravo\Documents\yemen music\features\features_30s.npy')
# labels = np.load(r'C:\Users\bravo\Documents\yemen music\features\labels_30s.npy')
# print(features.shape)
# print(labels.shape)

# # prompt: show the labels
# # pleases remove the last _10s from every labels 

# new_labels = []
# for label in labels:
#   new_label = label.replace('_10s', '')
#   new_labels.append(new_label)

# print(new_labels)


# import os
# import numpy as np
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import xgboost as xgb
# import logging

# # Load features and labels
# # features = np.load('/content/drive/MyDrive/yemeni_music/app/features_30s.npy')
# # labels = np.load('/content/drive/MyDrive/yemeni_music/app/labels_30s.npy')
# # print(features.shape)
# # print(labels.shape)

# # Ensure variable names are consistent
# X_combined = features
# y_combined = new_labels

# # Convert labels (music types) into a format usable by models
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y_combined)
# label_names = label_encoder.inverse_transform(np.unique(y_encoded))

# # Reshape X to be a 2D array (n_samples, n_features) for XGBoost
# if X_combined.ndim == 3:  # If X is 3D (e.g., [n_samples, n_time_steps, n_features]), reshape to 2D
#     n_samples, n_time_steps, n_features = X_combined.shape
#     X_combined = X_combined.reshape(n_samples, n_time_steps * n_features)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_combined, y_encoded, test_size=0.2, random_state=42)

# # Create a folder for saving results
# experiment_folder = "./results"
# if not os.path.exists(experiment_folder):
#     os.makedirs(experiment_folder)

# # Optional: You can implement a function to plot label distribution before and after undersampling
# # Example:
# plot_label_distribution(y_train, label_names, f'{experiment_folder}/before_undersampling')

# # Function for undersampling (define your own method based on the task)
# # For now, assuming a placeholder function
# def undersample_data(X, y):
#     # Placeholder: return data without any modification
#     return X, y

# # Perform undersampling to balance the dataset
# X_train, y_train = undersample_data(X_train, y_train)

# # Optional: Plot label distribution after undersampling
# plot_label_distribution(y_train, label_names, f'{experiment_folder}/after_undersampling')

# # Example XGBoost model
# xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)

# # Train and evaluate XGBoost model
# xgb_model.fit(X_train, y_train)
# xgb_score = xgb_model.score(X_test, y_test)
# print(f'XGBoost Model Accuracy: {xgb_score * 100:.2f}%')

# # Prepare data for deep learning models (reshape for 1D CNN)
# X_train_dl = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test_dl = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Number of epochs for training deep learning models
# epochs = 250 


# # Training different models
# VGG19_model = VGG19_1D(input_shape=(X_train_dl.shape[1], 1))
# VGG19_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# train_and_evaluate_dl_model(VGG19_model, 'VGG19', X_train_dl, X_test_dl, y_train, y_test, experiment_folder, label_names, epochs)

# #VGG16 Model for 1D data
# vgg16_1d_model = VGG16_1D(input_shape=(X_train_dl.shape[1], 1))
# vgg16_1d_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# train_and_evaluate_dl_model(vgg16_1d_model, 'VGG16', X_train_dl, X_test_dl, y_train, y_test, experiment_folder, label_names, epochs=epochs)

# # MobileNet Model for 1D data
# mobilenet_1d_model = mobilenet_1d(input_shape=(X_train_dl.shape[1], 1))
# mobilenet_1d_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# train_and_evaluate_dl_model(mobilenet_1d_model, 'Mobilenet', X_train_dl, X_test_dl, y_train, y_test, experiment_folder, label_names, epochs=epochs)

# # ResNet50 Model for 1D data
# ResNet50_1D_model = ResNet50_1D(input_shape=(X_train_dl.shape[1], 1))
# ResNet50_1D_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# train_and_evaluate_dl_model(ResNet50_1D_model, 'ResNet50', X_train_dl, X_test_dl, y_train, y_test, experiment_folder, label_names, epochs=epochs)

# # AlexNet Model for 1D data
# alexnet_1d_model = AlexNet_1D(input_shape=(X_train_dl.shape[1], 1))
# alexnet_1d_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# train_and_evaluate_dl_model(alexnet_1d_model, 'AlexNet', X_train_dl, X_test_dl, y_train, y_test, experiment_folder, label_names, epochs=epochs)


# Argument parser setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple machine learning and deep learning models on audio features.")
    
    # Add arguments
    parser.add_argument('--features', type=str, required=True, help="Path to the .npy file containing features.")
    parser.add_argument('--labels', type=str, required=True, help="Path to the .npy file containing labels.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save results.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the data to use as test set.")
    parser.add_argument('--epochs', type=int, default=250, help="Number of epochs for training deep learning models.")
    parser.add_argument('--patience', type=int, default=15, help="Patience for early stopping in deep learning models.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for deep learning models.")
    parser.add_argument('--n_estimators', type=int, default=100, help="Number of estimators for XGBoost.")
    parser.add_argument('--undersample', action='store_true', help="Whether to perform undersampling on the training data.")

    args = parser.parse_args()
    
    # Call the main function
    main(args)

# python training.py --features ./features/features_30s.npy --labels ./features/labels_30s.npy --output_dir ./results --epochs 300 --undersample
