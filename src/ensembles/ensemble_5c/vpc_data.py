import tensorflow as tf
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE = 30
BATCH_SIZE = 32



def load_training_data(validation_split=0.2):


    TRAIN_GRAY_DIR_PATH = "../../data/working/train_grayscale_normalized" 

    gray_image_data = []
    image_labels = []
    
    folders = os.listdir(TRAIN_GRAY_DIR_PATH)
    print(f"Loading Training data from {len(folders)} directories")
    for i in range(43):
        folder = folders[i]


        # Load grayscale images
        training_gray_data_path = f"{TRAIN_GRAY_DIR_PATH}/{folder}"
        training_gray_images = os.listdir(training_gray_data_path)
        for img in training_gray_images:
            path_to_img = f"{training_gray_data_path}/{img}"
            #print(f"Opening image {path_to_img}")
            try:
                image_fromarray = None
                image = cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)
                image_fromarray = Image.fromarray(image, 'L')

                resize_image = image_fromarray.resize((IMG_SIZE, IMG_SIZE))

                gray_image_data.append(np.array(resize_image))
                image_labels.append(i)

            except Exception as e:
                print("Error in " + img)
                print(e)

        

    print("Training images loaded. Shuffling data.")

    # Changing the list to numpy array
    gray_image_data = np.array(gray_image_data)
    image_labels = np.array(image_labels)

    ###############################
    shuffle_indexes = np.arange(gray_image_data.shape[0])
    np.random.shuffle(shuffle_indexes)
    gray_image_data = gray_image_data[shuffle_indexes]
    image_labels = image_labels[shuffle_indexes]


    ########################
    X_train_gray, X_val_gray, y_train, y_val = train_test_split(gray_image_data, image_labels, test_size=validation_split, random_state=42, shuffle=False)


    X_train_gray = X_train_gray/255 
    X_val_gray = X_val_gray/255

    # Grayscale arrays have to be reshaped
    X_train_gray = X_train_gray.reshape(X_train_gray.shape[0], IMG_SIZE, IMG_SIZE, 1)
    X_val_gray = X_val_gray.reshape(X_val_gray.shape[0], IMG_SIZE, IMG_SIZE, 1)

    print("X_train_gray.shape", X_train_gray.shape)
    print("X_valid_gray.shape", X_val_gray.shape)
    print("y_train.shape", y_train.shape)
    print("y_valid.shape", y_val.shape)


    #############
    y_train = tf.keras.utils.to_categorical(y_train, 43)
    y_val = tf.keras.utils.to_categorical(y_val, 43)


    return X_train_gray, y_train, X_val_gray, y_val



def load_testing_data():

    TEST_GRAY_DIR_PATH = "../../data/working/test_grayscale_normalized" 

    gray_image_data = []
    image_labels = []
    
    folders = os.listdir(TEST_GRAY_DIR_PATH)
    print(f"Loading testing data from {len(folders)} directories")
    for i in range(43):
        folder = folders[i]

    

        # Load grayscale images
        test_gray_data_path = f"{TEST_GRAY_DIR_PATH}/{folder}"
        test_gray_images = os.listdir(test_gray_data_path)
        for img in test_gray_images:
            path_to_img = f"{test_gray_data_path}/{img}"
            #print(f"Opening image {path_to_img}")
            try:
                image_fromarray = None
                image = cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)
                image_fromarray = Image.fromarray(image, 'L')

                resize_image = image_fromarray.resize((IMG_SIZE, IMG_SIZE))

                gray_image_data.append(np.array(resize_image))
                image_labels.append(i)

            except Exception as e:
                print("Error in " + img)
                print(e)

        

    print("Test images loaded.")

    # Changing the list to numpy array
    gray_image_data = np.array(gray_image_data)
    image_labels = np.array(image_labels)

    gray_image_data = gray_image_data / 255

    # Grayscale arrays have to be reshaped
    gray_image_data = gray_image_data.reshape(gray_image_data.shape[0], IMG_SIZE, IMG_SIZE, 1)

    print("test_gray data shape: ", gray_image_data.shape)
    print("test_labels shape: ", image_labels.shape)


    return gray_image_data, image_labels
