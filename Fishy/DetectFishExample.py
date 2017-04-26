#code from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

K.set_image_dim_ordering('th')


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = r'data\train'
validation_data_dir = r'data\validation'
trainSamples = 6000
validationSamples = 3000
batchSize=32
stepsPerEpoch=trainSamples/batchSize
validationPerEpoch=validationSamples/batchSize
epochs = 50


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, img_width, img_height)))
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
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
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
model.save_weights('first_try.h5')
