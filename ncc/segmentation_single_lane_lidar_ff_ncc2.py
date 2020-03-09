

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt
import os
import keras

from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, UpSampling2D, Input, Dense, concatenate, Add
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Softmax
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.optimizers import Adam

from dataloader import DataLoader

import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def import_data():
    # IMAGE_DIR_PATH = '../dataset/single_lane/training/image_2'
    # MASK_DIR_PATH = '../dataset/single_lane/training/gt_image_2'
    # LIDAR_DIR_PATH = '../dataset/single_lane/training/lidar_2d'

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
                                 augment = True,
                                 shuffle = True)

    train_images = []
    train_mask = []
    train_labels = []
    lidar_images = []

    for image, mask, lidar in dataset:
        train_images.append(image)
        train_mask.append(mask)
        lidar_images.append(lidar)
        # print(image.shape)

    #train_data = np.stack((train_images, lidar_images), axis = 1)
    print(np.shape(train_images))
    print(np.shape(train_mask))
    # print((train_images, train_mask))
    return train_images, train_mask, lidar_images

def import_test_data():
    # IMAGE_DIR_PATH = '../dataset/single_lane/testing/image'
    # MASK_DIR_PATH = '../dataset/single_lane/testing/masks'
    # LIDAR_DIR_PATH = '../dataset/single_lane/testing/lidar_2d'

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

    for image, mask, lidar in dataset:
        test_images.append(image)
        test_mask.append(mask)
        lidar_images.append(lidar)

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
    print('lol', np.shape(test[1][0]))
    sample_image, sample_mask, sample_lidar = test[0][0], test[1][0], test[2][0]

    #sample_image = train[0]
    #sample_mask = mask[0]
    print("lol")
    # display([sample_image, sample_mask, sample_lidar])
    print("displayed")
    #print(sample_image)
    return sample_image, sample_mask, sample_lidar

def hsi():
    pool_size = (2, 2)

    input = Input(shape=(128, 128, 3))
    x = Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu') (input)
    x = Conv2D(128, (2, 2), padding='same', strides=(1, 1), activation='relu') (x)
    x = MaxPooling2D(pool_size=pool_size) (x)

    #residual block A
    x1 = Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu') (x)
    x2 = Conv2D(32, (2, 2), padding='same', strides=(1, 1), activation='relu') (x1)
    x3 = Conv2D(32, (2, 2), padding='same', strides=(1, 1), activation='relu')(x2)

    x4 = concatenate([x1, x2, x3])
    x = Add()([x4, x])

    #residual block A
    x1 = Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu')(x)
    x2 = Conv2D(32, (2, 2), padding='same', strides=(1, 1), activation='relu')(x1)
    x3 = Conv2D(32, (2, 2), padding='same', strides=(1, 1), activation='relu')(x2)

    x4 = concatenate([x1, x2, x3])
    x = Add()([x4, x])

    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Conv2D(256, (2, 2), padding='same', strides=(1, 1), activation='relu') (x)

    #residual block B
    x1 = Conv2D(128, (2, 2), padding='same', strides=(1, 1), activation='relu')(x)
    x2 = Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu')(x1)
    x3 = Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu')(x2)

    x4 = concatenate([x1, x2, x3])
    x = Add()([x4, x])

    # residual block B
    x1 = Conv2D(128, (2, 2), padding='same', strides=(1, 1), activation='relu')(x)
    x2 = Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu')(x1)
    x3 = Conv2D(64, (2, 2), padding='same', strides=(1, 1), activation='relu')(x2)

    x4 = concatenate([x1, x2, x3])
    x = Add()([x4, x])
    x = UpSampling2D(size=(2, 2))(x)
    #x = GlobalAveragePooling2D(data_format='channels_last') (x)
    x = Dense(128, activation='relu') (x)
    x = Dense(64, activation='softmax') (x)

    model = Model(inputs=input, outputs=x)
    return model

def feature_fusion():
    model1 = hsi()
    model2 = hsi()

    x = concatenate([model1.output, model2.output])
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(3, (2, 2), padding='same', strides=(1, 1), activation='relu')(x)
    x = Conv2D(3, (2, 2), padding='same', strides=(1, 1), activation='relu')(x)
    x = Conv2D(3, (2, 2), padding='same', strides=(1, 1), activation='relu')(x)
    x = Conv2D(3, (2, 2), padding='same', strides=(1, 1), activation='relu')(x)
    x = Conv2D(3, (2, 2), padding='same', strides=(1, 1), activation='relu')(x)
    output = x

    model = Model(inputs=[model1.input, model2.input], outputs=output)

    return model
def create_model():
    OUTPUT_CHANNELS = 3
    #model = unet_model(OUTPUT_CHANNELS)
    model = feature_fusion()

    optimizer = keras.optimizers.Adam(lr=1e-6)
    model.compile(optimizer=optimizer, metrics=['accuracy'], loss='mean_squared_error')

    #model.summary()
    return model

def load_model():

    checkpoint_path = "training_single_lane_lidar_ff_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a new model instance
    model = create_model()

    # Load the previously saved weights
    model.load_weights('training_single_lane_lidar_ff_2/weights.h5')
    
    test_images, test_labels, lidar = import_test_data()
    test_images = test_images[0].numpy()
    test_labels = test_labels[0].numpy()
    lidar = lidar[0].numpy()

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
    sample_image = sample_image[0]
    sample_lidar = sample_lidar[0]
    sample_mask = sample_mask[0]
    #print(tf.reshape(sample_image, (1,128,128,3)))
    sample_image = tf.reshape(sample_image, (1,128,128,3))
    sample_lidar = tf.reshape(sample_lidar, (1, 128, 128, 3))
    sample_mask = tf.reshape(sample_mask, (1, 128, 128, 3))
    print(np.shape(sample_mask))
    if dataset:
        for image, mask in dataset.take(num):
          pred_mask = model.predict(image)
          display([image[0], mask[0], create_mask(pred_mask)])
    else:
        print(np.shape(sample_lidar))
        display([sample_image, sample_mask,
                 create_mask(model.predict([np.array(sample_image), np.array(sample_lidar)], steps=1))])

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def train(model):

    train_dataset, test_dataset, STEPS_PER_EPOCH = get_train()

    # test_image, test_mask, test_lidar = show_example()

    checkpoint_path = "training_single_lanes_lidar_ff_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #return
    #model = create_model()
    EPOCHS = 5
    VAL_SUBSPLITS = 5
    BATCH_SIZE = 200
    VALIDATION_STEPS = 100 // BATCH_SIZE // VAL_SUBSPLITS
    model_history = model.fit(x=[train_dataset[0][0], train_dataset[2][0]], y=train_dataset[1][0], epochs=EPOCHS,
                              #batch_size=BATCH_SIZE,
                              steps_per_epoch=100,
                              validation_steps=82,
                              validation_data=([test_dataset[0][0], test_dataset[2][0]], test_dataset[1][0]))
                              #callbacks=[cp_callback])

    model.save_weights('training_single_lane_lidar_ff_2/weights.h5')
    model.save('training_single_lane_lidar_ff_2/fcn.h5')

    model.summary()
    #show_predictions(model)



def main():
    # model = create_model()
    # keras.utils.plot_model(model, to_file='model.png')
    # model.summary()
    model = load_model()
    model.summary()
    #model = keras.models.load_model('training_lanes_fcn_1/fcn.h5')
    # train(model)
    show_predictions(model)

if __name__ == "__main__":
    main()
