# Arda Mavi

import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import boto3
import botocore


def save_model(model):
    if not os.path.exists('Data/Model/'):
        os.makedirs('Data/Model/')
    model_json = model.to_json()
    with open("Data/Model/model.json", "w") as model_file:
        model_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Data/Model/weights.h5")
    print('Model and weights saved')
    return


def save_s3_model():
    """
    Save training model to s3.
    """
    # get credentail
    ENDPOINT_URL = os.getenv('ENDPOINT_URL', None)
    ACCESS_KEY = os.getenv('ACCESS_KEY', None)
    SECRET_KEY = os.getenv('SECRET_KEY', None)
    # set variable
    BUCKET_NAME = 'train-model' # bucket name in s3

    # set s3 connection
    s3_client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
        verify=False,
        config=botocore.config.Config(signature_version='s3')
    )

    # check buckets exists
    buckets = s3_client.list_buckets()
    if BUCKET_NAME not in buckets:
        s3_client.create_bucket(Bucket=BUCKET_NAME)
        
    # save model
    s3_client.upload_file('Data/Model/weights.h5', BUCKET_NAME, 'weights.h5') # upload file to s3


def get_model(num_classes=2):
    # --- allow gpu memory growth ---
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # allow memory growth
    config.log_device_placement = False # do not show device info
    sess = tf.Session(config=config)
    set_session(sess)
    # ------

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    save_model(get_model())
