# Arda Mavi

import numpy as np
import os
from os import listdir
from skimage import io
from scipy.misc import imresize
from keras.preprocessing.image import array_to_img, img_to_array, load_img

import boto3
import botocore


def get_img(data_path):
    # Getting image array from path:
    img_size = 64
    img = io.imread(data_path)
    img = imresize(img, (img_size, img_size, 3))
    return img


def get_s3_file():
    """
    Get training data from s3.
    """
    # get credentail
    ENDPOINT_URL = os.getenv('ENDPOINT_URL', None)
    ACCESS_KEY = os.getenv('ACCESS_KEY', None)
    SECRET_KEY = os.getenv('SECRET_KEY', None)
    # set variable
    BUCKET_NAME = 'npy-file' # bucket name in s3

    # set s3 connection
    s3_client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
        config=botocore.config.Config(signature_version='s3')
    )

    # # list s3 buckets
    # resp = s3_client.list_buckets()
    # print(resp['Buckets'])

    # download file
    if not os.path.exists('Data/npy_train_data/'):
        os.makedirs('Data/npy_train_data/')
    
    s3_client.download_file(BUCKET_NAME, 'X.npy', 'Data/npy_train_data/X.npy')   # download X npy file
    s3_client.download_file(BUCKET_NAME, 'Y.npy', 'Data/npy_train_data/Y.npy')   # download Y npy file


def get_dataset(dataset_path='Data/Train_Data'):
    get_s3_file()   # get training data

    # Getting all data from data path:
    try:
        X = np.load('Data/npy_train_data/X.npy')
        Y = np.load('Data/npy_train_data/Y.npy')
    except:
        labels = listdir(dataset_path) # Geting labels
        print('Categories:\n', labels)
        len_datas = 0
        for label in labels:
            len_datas += len(listdir(dataset_path+'/'+label))
        X = np.zeros((len_datas, 64, 64, 3), dtype='float64')
        Y = np.zeros(len_datas)
        count_data = 0
        count_categori = [-1,''] # For encode labels
        for label in labels:
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = get_img(datas_path+'/'+data)
                X[count_data] = img
                # For encode labels:
                if label != count_categori[1]:
                    count_categori[0] += 1
                    count_categori[1] = label
                Y[count_data] = count_categori[0]
                count_data += 1
        # Create dateset:
        import keras
        Y = keras.utils.to_categorical(Y)
        import os
        if not os.path.exists('Data/npy_train_data/'):
            os.makedirs('Data/npy_train_data/')
        np.save('Data/npy_train_data/X.npy', X)
        np.save('Data/npy_train_data/Y.npy', Y)
    X /= 255.
    from sklearn.model_selection import train_test_split
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X, X_test, Y, Y_test
