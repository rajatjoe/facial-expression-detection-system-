import os
import tensorflow as tf
import numpy as np
import shutil
from pathlib import Path

# Fix for the integer overflow error in TensorFlow's signbit function
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops

# Patch TensorFlow's numpy implementation to avoid overflow
def apply_signbit_patch():
    try:
        # Import the module containing the problematic function
        from keras.src.backend.tensorflow import numpy as tf_numpy
        
        # Create a replacement function that avoids the overflow
        original_signbit = tf_numpy.signbit
        def safe_signbit(x):
            try:
                # For floating point types, use the original implementation
                if tf.is_tensor(x) and x.dtype.is_floating:
                    return tf.math.less(x, tf.constant(0, dtype=x.dtype))
                # For integer types, use a different approach
                else:
                    # Convert to float32 first to avoid overflow
                    x_float = tf.cast(x, tf.float32)
                    return tf.math.less(x_float, tf.constant(0, dtype=tf.float32))
            except Exception as e:
                print(f"Error in safe_signbit: {e}")
                # Fallback to a very simple implementation
                return tf.math.less(tf.cast(x, tf.float32), 0)
        
        # Replace the function
        tf_numpy.signbit = safe_signbit
        print("Successfully patched TensorFlow signbit function")
    except Exception as e:
        print(f"Failed to patch signbit function: {e}")
        # Try an alternative approach
        try:
            # Monkey patch at a lower level
            import keras.src.backend.tensorflow.numpy
            keras.src.backend.tensorflow.numpy.signbit = lambda x: tf.math.less(tf.cast(x, tf.float32), 0)
            print("Applied alternative signbit patch")
        except Exception as e2:
            print(f"Alternative patch also failed: {e2}")

# Apply the patch
apply_signbit_patch()

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet import ResNet101
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from matplotlib import pyplot as plt
from tensorflow import keras

from common_functions import (
    create_model,
    get_data,
    fit,
    evaluation_model,
    saveModel,
)

# Function to create a sample dataset with 10 images per category
def create_sample_dataset(source_path, target_path, samples_per_class=10):
    # Create target directory if it doesn't exist
    os.makedirs(target_path, exist_ok=True)
    
    # Get all emotion categories
    categories = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
    
    for category in categories:
        source_category_path = os.path.join(source_path, category)
        target_category_path = os.path.join(target_path, category)
        
        # Create category directory in target
        os.makedirs(target_category_path, exist_ok=True)
        
        # Get all image files in the category
        image_files = [f for f in os.listdir(source_category_path) 
                      if os.path.isfile(os.path.join(source_category_path, f)) 
                      and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Select 10 random images (or all if less than 10)
        selected_files = image_files[:min(samples_per_class, len(image_files))]
        
        print(f"Copying {len(selected_files)} images for category {category}")
        
        # Copy selected files to target directory
        for file in selected_files:
            source_file = os.path.join(source_category_path, file)
            target_file = os.path.join(target_category_path, file)
            shutil.copy2(source_file, target_file)
    
    return target_path

if __name__ == "__main__":
    # Define base paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # Create trained_models directory if it doesn't exist
    trained_models_dir = os.path.join(base_dir, "trained_models")
    os.makedirs(trained_models_dir, exist_ok=True)
    
    # Set memory growth to avoid OOM errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Memory growth enabled for {device}")
            except:
                print("Memory growth setting failed")
    
    # Original dataset paths
    original_train_path = os.path.join(base_dir, "Databases", "FER-2013", "train")
    original_test_path = os.path.join(base_dir, "Databases", "FER-2013", "test")
    
    # Sample dataset paths
    sample_train_path = os.path.join(base_dir, "Databases", "FER-2013-Sample", "train")
    sample_test_path = os.path.join(base_dir, "Databases", "FER-2013-Sample", "test")
    
    # Create sample datasets with 10 images per class
    print("Creating sample training dataset...")
    create_sample_dataset(original_train_path, sample_train_path, samples_per_class=10)
    print("Creating sample testing dataset...")
    create_sample_dataset(original_test_path, sample_test_path, samples_per_class=10)
    
    parameters = {
        "shape": [80, 80],
        "nbr_classes": 7,
        "train_path": sample_train_path,  # Use sample dataset
        "test_path": sample_test_path,    # Use sample dataset
        "batch_size": 4,                  # Smaller batch size for small dataset
        "epochs": 10,                     # Fewer epochs for testing
        "number_of_last_layers_trainable": 5,  # Fewer trainable layers
        "learning_rate": 0.001,
        "nesterov": True,
        "momentum": 0.9,
    }
    
    # Add this to use mixed precision training which can help with memory issues
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Using mixed precision training")
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(current_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    model, filename, preprocess_input = None, None, None

    choice = input(
        "which models do you want to train?"
        "\n\t-1- resnet50"
        "\n\t-2- vgg16"
        "\n\t-3- xception"
        "\n\t-4- inception_resnet_v2"
        "\n\t-5- inception_v3"
        "\n\t-6- resnet50v2"
        "\n\t-7- resnet101"
        "\n>>>"
    )
    
    # Rest of the code remains the same
    if choice == "1":
        model = ResNet50
        preprocess_input = keras.applications.resnet.preprocess_input
        filename = "resnet50"
    elif choice == "2":
        model = VGG16
        filename = "vgg16"
        preprocess_input = keras.applications.vgg16.preprocess_input
    elif choice == "3":
        model = Xception
        filename = "xception"
        preprocess_input = keras.applications.xception.preprocess_input
    elif choice == "4":
        model = InceptionResNetV2
        filename = "inception_resnet_v2"
        preprocess_input = keras.applications.inception_resnet_v2.preprocess_input
    elif choice == "5":
        model = InceptionV3
        filename = "inception_v3"
        preprocess_input = keras.applications.inception_v3.preprocess_input
    elif choice == "6":
        model = ResNet50V2
        filename = "resnet50v2"
        preprocess_input = keras.applications.resnet_v2.preprocess_input
    elif choice == "7":
        model = ResNet101
        filename = "resnet101"
        preprocess_input = keras.applications.resnet.preprocess_input
    else:
        print("you have to choose a number between 1 and 7")
        exit(1)

    if model is not None and filename is not None:
        print("Training parameters:")
        for key, value in parameters.items():
            print(f"{key}: {value}")

        # Verify dataset paths exist
        if not os.path.exists(parameters["train_path"]) or not os.path.exists(parameters["test_path"]):
            print("Error: Dataset paths not found. Please ensure the FER-2013 dataset is properly set up.")
            exit(1)

        train_files, test_files, train_generator, test_generator = get_data(
            preprocess_input=preprocess_input, parameters=parameters
        )

        model = create_model(architecture=model, parameters=parameters)

        history = fit(
            model=model,
            train_generator=train_generator,
            test_generator=test_generator,
            train_files=train_files,
            test_files=test_files,
            parameters=parameters,
        )

        score = evaluation_model(model, test_generator)

        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label="Training Accuracy")
        plt.plot(val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.ylabel("Accuracy")
        plt.ylim([min(plt.ylim()), 1])
        plt.title("Training and Validation Accuracy")

        plt.subplot(2, 1, 2)
        plt.plot(loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.ylabel("Cross Entropy")
        plt.ylim([0, 1.0])
        plt.title("Training and Validation Loss")
        plt.xlabel("epoch")
        plt.show()

        filename = f"{filename}_ferplus"
        log_file = os.path.join(logs_dir, f"{filename}_parameters.log")

        if os.path.isfile(log_file):
            with open(log_file, "r") as file:
                print("\nPrevious training parameters:")
                print(file.read())

        choice = input("\nSave model? (Y/N)\n>>>")

        if choice.upper() == "Y":
            # Pass the trained_models_dir to saveModel function
            saveModel(filename=filename, model=model, models_dir=trained_models_dir)
            with open(log_file, "w") as file:
                file.write(f"Parameters:\n")
                for key, value in parameters.items():
                    file.write(f"{key}: {value}\n")
                file.write(f"\nValidation Accuracy: {val_acc[-1]:.4f}")
                file.write(f"\nValidation Loss: {val_loss[-1]:.4f}")
                file.write(f"\nModel saved in: {trained_models_dir}")
