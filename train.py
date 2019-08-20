# Arda Mavi

import os
from keras.callbacks import ModelCheckpoint, TensorBoard
import requests
from get_model import save_s3_model


def train_model(model, X, X_test, Y, Y_test):
    checkpoints = []
    if not os.path.exists('Data/Checkpoints/'):
        os.makedirs('Data/Checkpoints/')
    checkpoints.append(ModelCheckpoint('Data/Checkpoints/best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
    checkpoints.append(TensorBoard(log_dir='Data/Checkpoints/./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

    # Creates live data:
    # For better yield. The duration of the training is extended.

    # If you don't want, use this:
    # model.fit(X, Y, batch_size=10, epochs=25, validation_data=(X_test, Y_test), shuffle=True, callbacks=checkpoints)

    from keras.preprocessing.image import ImageDataGenerator
    generated_data = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,  width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True, vertical_flip = False)
    generated_data.fit(X)
    import numpy
    model.fit_generator(generated_data.flow(X, Y, batch_size=8), steps_per_epoch=X.shape[0]//8, epochs=25, validation_data=(X_test, Y_test), callbacks=checkpoints)

    return model


def main():
    # get record log param
    API_LOG_HOST = os.getenv('API_LOG_HOST', None)
    API_LOG_PORT = os.getenv('API_LOG_PORT', None)
    API_LOG_JOB_NAME = os.getenv('API_LOG_JOB_NAME', None)
    req_log_url = 'http://{host}:{port}/jobs/{job_name}'.format(
                    host=API_LOG_HOST,
                    port=API_LOG_PORT,
                    job_name=API_LOG_JOB_NAME
                )

    # get dataset
    from get_dataset import get_dataset
    result = requests.post(url=req_log_url,
                json={"exec_stage":"get_dataset", "exec_status":"start"},
                verify=False
            )
    X, X_test, Y, Y_test = get_dataset()
    result = requests.post(url=req_log_url,
                json={"exec_stage":"get_dataset", "exec_status":"finish"},
                verify=False
            )

    # set model
    from get_model import get_model, save_model
    model = get_model(len(Y[0]))
    
    # training
    import numpy
    result = requests.post(url=req_log_url,
                json={"exec_stage":"training", "exec_status":"start"},
                verify=False
            )
    model = train_model(model, X, X_test, Y, Y_test)
    result = requests.post(url=req_log_url,
                json={"exec_stage":"training", "exec_status":"finish"},
                verify=False
            )

    # save model
    result = requests.post(url=req_log_url,
                json={"exec_stage":"save_model", "exec_status":"start"},
                verify=False
            )
    save_model(model)
    save_s3_model() # save model to s3
    result = requests.post(url=req_log_url,
                json={"exec_stage":"save_model", "exec_status":"finish"},
                verify=False
            )

    return model


if __name__ == '__main__':
    # use CPU
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    main()
