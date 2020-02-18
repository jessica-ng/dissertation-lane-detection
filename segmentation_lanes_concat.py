

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt
import os
import keras
import segmentation_lanes_fcn as fcn
import segmentation_lanes as unet

from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D, concatenate
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import *
from keras.applications.imagenet_utils import *
from keras.applications.vgg16 import *
import keras.backend as K
from keras.models import Model

from dataloader import DataLoader

import numpy as np

from utils.BilinearUpSampling import *
from utils.resnet_helpers import *


def import_data():
    IMAGE_DIR_PATH = '/home/jessica/Downloads/currentData/training/image_2'
    MASK_DIR_PATH = '/home/jessica/Downloads/currentData/training/semantic_rgb'

    # create list of PATHS
    image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
    mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

    dataset = DataLoader(image_paths=image_paths,
                         mask_paths=mask_paths,
                         image_size=[128, 128],
                         crop_percent=None,
                         channels=[3, 3],
                         seed=47)



    dataset = dataset.data_batch(batch_size=1000,
                                 augment = True,
                                 shuffle = True)

    train_images = []
    train_mask = []

    for image, mask in dataset:
        train_images.append(image)
        train_mask.append(mask)

    # print((train_images, train_mask))
    return [train_images, train_mask]

def import_test_data():
    IMAGE_DIR_PATH = '/home/jessica/Downloads/currentData/testing/image'
    MASK_DIR_PATH = '/home/jessica/Downloads/currentData/testing/mask'

    # create list of PATHS
    image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
    mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

    dataset = DataLoader(image_paths=image_paths,
                         mask_paths=mask_paths,
                         image_size=[128, 128],
                         crop_percent=None,
                         channels=[3, 3],
                         seed=47)

    dataset = dataset.data_batch(batch_size=100,
                                 augment=True,
                                 shuffle=True)

    test_images = []
    test_mask = []

    for image, mask in dataset:
        test_images.append(image)
        test_mask.append(mask)

    return [test_images, test_mask]

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask


def get_train():
    TRAIN_LENGTH = 100
    BATCH_SIZE = 100
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    # train = dataset['train'].map(train_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # test = dataset['test'].map(test_dataset)

    train_dataset = import_data()
    test_dataset = import_test_data()

    # train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    # train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # test_dataset = test.batch(BATCH_SIZE)
    print("length", len(train_dataset[1][0]))

    return train_dataset, test_dataset, STEPS_PER_EPOCH

def get_train2():
    TRAIN_LENGTH = 100
    BATCH_SIZE = 100
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    # train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # test = dataset['test'].map(load_image_test)

    train, mask = import_data()
    test = import_test_data()

    # train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    # train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # test_dataset = test.batch(BATCH_SIZE)

    return train, mask

def display(display_list):

    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(2):
      plt.subplot(1, len(display_list), i+1)
      plt.title(title[i])
      plt.imshow(display_list[i][0])
      plt.axis('off')

    if len(display_list) > 2:
        plt.subplot(1, len(display_list), 3)
        plt.title(title[2])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[2]))
        plt.axis('off')
    plt.show()
    # print(display_list[0])
    #
    # cv2.imshow("Display window", display_list[0])
    # cv2.imshow("Display window", display_list[1])
    #
    # cv2.waitkey(0)

def show_example():
    train, test, a= get_train()

    sample_image, sample_mask = train[0][0], train[1][0]

    #sample_image = train[0]
    #sample_mask = mask[0]

    display([sample_image, sample_mask])

    return sample_image, sample_mask

def resnet_model(output_channels):

    img_input = Input(shape=[128,128,3])
    image_size = [128, 128, 3]

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(0))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b')(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c')(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d')(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f')(x)

    x = conv_block(3, [512, 512, 2048], stage=5, block='a')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='b')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='c')(x)
    #classifying layer
    x = Conv2D(3, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(0))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)

    model = keras.models.Model(img_input, x)
    return model

def create_model():
    OUTPUT_CHANNELS = 3
    model = resnet_model(OUTPUT_CHANNELS)


    model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

    tf.keras.utils.plot_model(model, show_shapes=True)
    #model.summary()
    return model

def load_model():

    checkpoint_path = "training_lanes_concat/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #latest = tf.train.latest_checkpoint(checkpoint_dir)

    # Create a new model instance
    model = concat()

    # Load the previously saved weights
    model.load_weights('training_lanes_concat_2/weights.h5')

    test_images, test_labels = import_test_data()

    test_images = test_images[0].numpy()
    test_labels = test_labels[0].numpy()

    # print(type(test_images))
    # # Re-evaluate the model
    loss, acc = model.evaluate([test_images, test_images], test_labels, verbose=2)
    print("yay")
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    return model


def create_mask(pred_mask):
    #print("mask", pred_mask)
    # pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask = pred_mask[..., tf.newaxis]
    print(pred_mask)
    return pred_mask[0]

def show_predictions( model, dataset=None, num=1):
    sample_image, sample_mask = show_example()
    #print("reshaped", sample_image[0,:,:,:])
    sample_image = sample_image[0,:,:,:]
    #print(tf.reshape(sample_image, (1,128,128,3)))
    sample_image = tf.reshape(sample_image, (1,128,128,3))
    if dataset:
        for image, mask in dataset.take(num):
          pred_mask = model.predict(image)
          display([image[0], mask[0], create_mask(pred_mask)])
    else:

        display([sample_image, sample_mask,
                 create_mask(model.predict([sample_image, sample_image], steps=1))])

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def train(model):

    train_dataset, test_dataset, STEPS_PER_EPOCH = get_train()

    test_image, test_mask = show_example()

    # loss, acc = model.evaluate(test_dataset)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    # checkpoint_path = "training_lanes_fcn_1/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)

    #print(train_dataset[0][0][0])

    #model = create_model()
    EPOCHS = 2
    VAL_SUBSPLITS = 5
    BATCH_SIZE = 100
    VALIDATION_STEPS = 100 // BATCH_SIZE // VAL_SUBSPLITS

    model_history = model.fit([train_dataset[0][0], train_dataset[0][0]], train_dataset[1][0], epochs=EPOCHS,
                              #batch_size=BATCH_SIZE,
                              steps_per_epoch=100,
                              validation_steps=100,
                              validation_data=([test_dataset[0][0], test_dataset[0][0]], test_dataset[1][0]))
                              #callbacks=[cp_callback])

    # model.fit_generator((train_dataset[0][0], train_dataset[1][0]),
    #                     #steps_per_epoch=STEPS_PER_EPOCH,
    #                     epochs=EPOCHS, verbose=0, validation_data=(test_dataset[0][0], test_dataset[1][0]))
    model.save_weights('training_lanes_concat_2/weights.h5')
    model.save('training_lanes_concat_2/concat.h5')

    model.summary()
    show_predictions(model)

def concat():
    fcn_model = fcn.create_model()
    #fcn_model = Model((128,128,3), fcn_model)
    unet_model = create_model()
    #unet_model = Model((128, 128, 3), unet_model)

    combined_model = concatenate([fcn_model.output, unet_model.output])

    x = Dense(3, activation="relu")(combined_model)
    x = Dense(3, activation="linear")(x)

    model = Model(inputs=[fcn_model.input, unet_model.input], outputs=x)

    #adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

    model.summary()

    tf.keras.utils.plot_model(model, show_shapes=True)

    return model

def main():
    model = concat()
    #model = load_model()
    #model = keras.models.load_model('training_lanes_resnet_1/resnet.h5')
    train(model)
    show_predictions(model)

if __name__ == "__main__":
    main()