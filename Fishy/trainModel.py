#code from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from VGG16 import VGG_16

K.set_image_dim_ordering('th')

############################
# paths and constants
############################
img_width, img_height = 150, 150
train_data_dir = r'data\train'
validation_data_dir = r'data\validation'
VGG16Weights = 'vgg16_weights.h5'
savingPath = 'simpleModelOnTop.h5'

############################
# parameters of the training
############################
trainSamples = 6000
validationSamples = 3000
batchSize=32
epochs = 50
stepsPerEpoch=trainSamples/batchSize
validationPerEpoch=validationSamples/batchSize
inputShape = (3,150,150)

def trainModel(model):


    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batchSize,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batchSize,
            class_mode='binary')

    model.fit_generator(
            train_generator,
            steps_per_epoch=stepsPerEpoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validationPerEpoch)

    #model.load_weights('first_try.h5')
    model.save_weights(VGG16Weights)

model = VGG_16(inputShape)
trainModel(model)
