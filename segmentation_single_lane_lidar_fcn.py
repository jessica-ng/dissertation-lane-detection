

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt
import os
import keras

from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, UpSampling2D, Input, Dense, concatenate
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.optimizers import Adam

from dataloader import DataLoader

import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import cv2


def import_data():
    IMAGE_DIR_PATH = '/home/jessica/Downloads/data_road/training/image_2'
    MASK_DIR_PATH = '/home/jessica/Downloads/data_road/training/gt_image_2'
    LIDAR_DIR_PATH = '/home/jessica/Downloads/data_road/training/lidar_2d'

    # create list of PATHS
    image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
    mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]
    lidar_paths = [os.path.join(LIDAR_DIR_PATH, x) for x in os.listdir(LIDAR_DIR_PATH) if x.endswith('.png')]

    dataset = DataLoader(image_paths=image_paths,
                         mask_paths=mask_paths,
                         label_paths=lidar_paths,
                         image_size=[128, 128],
                         crop_percent=None,
                         channels=[3, 3],
                         seed=47,
                         lidar=True)



    dataset = dataset.data_batch(batch_size=100,
                                 augment = False,
                                 shuffle = False)

    train_images = []
    train_mask = []
    train_labels = []
    lidar_images = []

    for image, mask, lidar in dataset.take(1):
        train_images.append(image)
        train_mask.append(mask)
        lidar_images.append(lidar)
        # print(image.shape)

    #train_data = np.stack((train_images, lidar_images), axis = 1)
    #print(train_data.shape)
    # print((train_images, train_mask))
    return train_images, train_mask, lidar_images

def import_test_data():
    IMAGE_DIR_PATH = '/home/jessica/Downloads/data_road/testing/image'
    MASK_DIR_PATH = '/home/jessica/Downloads/data_road/testing/masks'
    LIDAR_DIR_PATH = '/home/jessica/Downloads/data_road/testing/lidar_2d'

    # create list of PATHS
    image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
    mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]
    lidar_paths = [os.path.join(LIDAR_DIR_PATH, x) for x in os.listdir(LIDAR_DIR_PATH) if x.endswith('.png')]


    dataset = DataLoader(image_paths=image_paths,
                         mask_paths=mask_paths,
                         label_paths=lidar_paths,
                         image_size=[128, 128],
                         crop_percent=None,
                         channels=[3, 3],
                         seed=47,
                         lidar = True)

    dataset = dataset.data_batch(batch_size=100,
                                 augment=False,
                                 shuffle=False)

    test_images = []
    test_mask = []
    test_labels = []
    lidar_images = []

    for image, mask, lidar in dataset.take(1):
        test_images.append(image)
        test_mask.append(mask)
        lidar_images.append(lidar)

    #test_data = np.stack((test_images, lidar_images), axis=1)
    print('test', len(test_images), len(test_images[0]))
    #exit()
    return test_images, test_mask, lidar_images

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
        print(display_list[2])
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
    train, test, lidar = get_train()
    print('lol', len(test[1][0]))
    sample_image, sample_mask, sample_lidar = test[0][0], test[1][0], test[2][0][0]

    #sample_image = train[0]
    #sample_mask = mask[0]
    print("lol")
    display([sample_image, sample_mask, sample_lidar])
    print("displayed")
    #print(sample_image)
    return sample_image, sample_mask, sample_lidar

def unet_model(output_channels):
    batch_size = 200
    epochs = 10
    pool_size = (2, 2)
    input_shape = [2, 128, 128, 3]
    inputA = Input(shape=(128,128,3))
    inputB = Input(shape=(128,128,3))

    # the first branch operates on the first input
    x = Dense(8, activation="relu")(inputA)
    x = Dense(4, activation="relu")(x)
    x = Model(inputs=inputA, outputs=x)
    # the second branch opreates on the second input
    y = Dense(64, activation="relu")(inputB)
    y = Dense(32, activation="relu")(y)
    y = Dense(4, activation="relu")(y)
    y = Model(inputs=inputB, outputs=y)
    # combine the output of the two branches
    combined = concatenate([x.output, y.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(3, activation="relu")(combined)
    #z = Dense(1, activation="linear")(z)
    # our model will accept the inputs of the two branches and
    # then output a single value

    # model = Model(inputs=[x.input, y.input], outputs=z)

    ### Here is the actual neural network ###
    # model = Sequential()

    # inputA =
    # # Normalizes incoming inputs. First layer needs the input shape to work
    # model.add(BatchNormalization(input_shape=input_shape))

    # Below layers were re-named for easier reading of model summary; this not necessary
    # Conv Layer 1
    z = Conv2D(8, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv1') (z)

    # Conv Layer 2
    z = Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv2') (z)

    # Pooling 1
    z = MaxPooling2D(pool_size=pool_size)(z)

    # Conv Layer 3
    z = Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv3')(z)
    z = Dropout(0.2)(z)

    # Conv Layer 4
    z = Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv4')(z)
    z = Dropout(0.2)(z)

    # Conv Layer 5
    z = Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv5')(z)
    z = Dropout(0.2)(z)

    # Pooling 2
    z = MaxPooling2D(pool_size=pool_size)(z)

    # Conv Layer 6
    z = Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv6') (z)
    z = Dropout(0.2)(z)

    # Conv Layer 7
    z = Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv7')(z)
    z = Dropout(0.2)(z)

    # Pooling 3
    z = MaxPooling2D(pool_size=pool_size)(z)

    # Upsample 1
    z = UpSampling2D(size=pool_size)(z)

    # Deconv 1
    z = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv1')(z)
    z = Dropout(0.2)(z)

    # Deconv 2
    z = Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv2')(z)
    z = Dropout(0.2)(z)

    # Upsample 2
    z = UpSampling2D(size=pool_size)(z)

    # Deconv 3
    z = Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv3')(z)
    z = Dropout(0.2)(z)

    # Deconv 4
    z = Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv4')(z)
    z = Dropout(0.2)(z)

    # Deconv 5
    z = Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv5')(z)
    z = Dropout(0.2)(z)

    # Upsample 3
    z = UpSampling2D(size=pool_size)(z)

    # Deconv 6
    z = Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv6')(z)

    # Final layer - only including one channel so 1 filter
    z = Conv2DTranspose(3, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Final')(z)

    model = Model(inputs=[x.input, y.input], outputs=z)


    return model

def create_model():
    OUTPUT_CHANNELS = 3
    model = unet_model(OUTPUT_CHANNELS)


    model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')

    tf.keras.utils.plot_model(model, show_shapes=True)
    #model.summary()
    return model

def load_model():

    checkpoint_path = "training_single_lane_lidar_fcn_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #latest = tf.train.latest_checkpoint(checkpoint_dir)

    # Create a new model instance
    model = create_model()

    # Load the previously saved weights
    model.load_weights('training_single_lane_lidar_fcn_1/weights.h5')

    test_images, test_labels, lidar = import_test_data()
    print(len(test_images), len(test_images[0]))
    test_images = test_images[0].numpy()
    test_labels = test_labels[0].numpy()
    lidar = lidar[0].numpy()

    print(np.shape(test_images))
    print(np.shape(test_labels))
    print(np.shape(lidar))
    # # Re-evaluate the model
    loss, acc = model.evaluate([np.array(test_images), np.array(lidar)], test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    return model


def create_mask(pred_mask):
    #print("mask", pred_mask)
    # pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask = pred_mask[..., tf.newaxis]
    #print(pred_mask)
    #print(pred_mask)
    return pred_mask[0]

def show_predictions( model, dataset=None, num=1):
    sample_image, sample_mask, sample_lidar = show_example()
    #print("reshaped", sample_image[0,:,:,:])
    sample_image = sample_image[0,:,:,:]
    #print(tf.reshape(sample_image, (1,128,128,3)))
    sample_image = tf.reshape(sample_image, (1,128,128,3))
    if dataset:
        for image, mask in dataset.take(num):
          pred_mask = model.predict(image)
          display([image[0], mask[0], create_mask(pred_mask)])
    else:
        print(sample_image.shape)
        sample_lidar = [sample_lidar]
        display([sample_image, sample_mask,
                 create_mask(model.predict([np.array(sample_image), np.array(sample_lidar)], steps=1))])

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def train(model):

    train_dataset, test_dataset, STEPS_PER_EPOCH = get_train()

    test_image, test_mask, test_lidar = show_example()

    # loss, acc = model.evaluate(test_dataset)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    checkpoint_path = "training_single_lanes_lidar_fcn_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)

    #return
    #model = create_model()
    EPOCHS = 1
    VAL_SUBSPLITS = 5
    BATCH_SIZE = 200
    VALIDATION_STEPS = 100 // BATCH_SIZE // VAL_SUBSPLITS


    print(len(train_dataset[2][0]), len(train_dataset[1][0]), len(train_dataset[0][0]))
    print(len(train_dataset[2][0][0]), len(train_dataset[1][0][0]))
    print(len(train_dataset[2][0][0][0]), len(train_dataset[1][0][0][0]))
    print(len(train_dataset[2][0][0][0][0]), len(train_dataset[1][0][0][0][0]))
    print(len(test_dataset[1][0]), len(test_dataset[0][0]), len(test_dataset[2][0]))
    model_history = model.fit([train_dataset[0][0], train_dataset[2][0]], train_dataset[1][0], epochs=EPOCHS,
                              #batch_size=BATCH_SIZE,
                              steps_per_epoch=100,
                              validation_steps=100,
                              validation_data=([test_dataset[0][0], test_dataset[2][0]], test_dataset[1][0]))
                              #callbacks=[cp_callback])

    # model.fit_generator((train_dataset[0][0], train_dataset[1][0]),
    #                     #steps_per_epoch=STEPS_PER_EPOCH,
    #                     epochs=EPOCHS, verbose=0, validation_data=(test_dataset[0][0], test_dataset[1][0]))
    model.save_weights('training_single_lane_lidar_fcn_1/weights.h5')
    model.save('training_single_lane_lidar_fcn_1/fcn.h5')

    model.summary()
    show_predictions(model)



def main():
    # model = create_model()
    # model.summary()
    model = load_model()
    #model.summary()
    #model = keras.models.load_model('training_lanes_fcn_1/fcn.h5')
    train(model)
    show_predictions(model)

if __name__ == "__main__":
    main()