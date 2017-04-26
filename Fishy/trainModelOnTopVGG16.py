#code from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.applications import vgg16

#K.set_image_dim_ordering('th')

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

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(150,150,3))

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batchSize,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, trainSamples // batchSize)
    np.save('bottleneck_features_train.npy',
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batchSize,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, validationSamples // batchSize)
    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batchSize,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


save_bottleneck_features()
train_top_model()
