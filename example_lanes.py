from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os

import numpy as np
from dataloader import DataLoader

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def import_data():
    IMAGE_DIR_PATH = '/home/jessica/Downloads/training/image_2'
    MASK_DIR_PATH = '/home/jessica/Downloads/training/semantic_rgb'

    # create list of PATHS
    image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
    mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

    dataset = DataLoader(image_paths=image_paths,
                         mask_paths=mask_paths,
                         image_size=[375, 1242],
                         crop_percent=None,
                         channels=[3, 3],
                         seed=47)



    dataset = dataset.data_batch(batch_size=1,
                                 augment = False,
                                 shuffle = False)

    train_images = []
    train_mask = []

    for image, mask in dataset:
        train_images.append(image)
        train_mask.append(mask)
    print(train_images[0])

    # print((train_images, train_mask))
    return train_images, train_mask

def import_test_data():
    IMAGE_DIR_PATH = '/home/jessica/Downloads/testing/image_2'
    # MASK_DIR_PATH = '/home/jessica/Downloads/training/semantic_rgb'

    # create list of PATHS
    image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
    # mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

    dataset = DataLoader(image_paths=image_paths,
                         # mask_paths=mask_paths,
                         image_size=[375, 1242],
                         crop_percent=None,
                         channels=[3, 3],
                         seed=47)

    dataset = dataset.data_batch(batch_size=1,
                                 augment=True,
                                 shuffle=True)

    # train_images = []
    # train_mask = []
    #
    # for image, mask in dataset:
    #     train_images.append(image)
    #     train_mask.append(mask)

    # print((train_images, train_mask))
    return dataset

def create_model():
    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, train_labels = import_data()

    #print(train_images)

    # Normalize pixel values to be between 0 and 1
    # train_images, test_images = train_images / 255.0, test_images / 255.0
    # train_images = np.float32(train_images) / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(375, (3, 3), activation='relu', input_shape=(375, 1242, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model



def train():
    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    #
    # # Normalize pixel values to be between 0 and 1
    # train_images, test_images = train_images / 255.0, test_images / 255.0
    #
    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    #                'dog', 'frog', 'horse', 'ship', 'truck']

    train_images, train_labels = import_data()

    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     # The CIFAR labels happen to be arrays,
    #     # which is why you need the extra index
    #     plt.xlabel(class_names[train_labels[i][0]])
    # plt.show()

    model = create_model()

    history = model.fit(train_images[0], train_labels[0], epochs=1)
                        #validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    #
    # print(test_acc)

    checkpoint_path = "training_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

def test():
    checkpoint_path = "training_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    latest = tf.train.latest_checkpoint(checkpoint_dir)

    # Create a new model instance
    model = create_model()

    # Load the previously saved weights
    model.load_weights(latest)

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Re-evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    heckpoint_path = "training_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels),
                        callbacks=[cp_callback])

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(test_acc)

def loadModel():

    checkpoint_path = "training_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    latest = tf.train.latest_checkpoint(checkpoint_dir)

    # Create a new model instance
    model = create_model()

    # Load the previously saved weights
    model.load_weights(latest)

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Re-evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    return model

def predict():
    model = loadModel()

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    predictions = model.predict(test_images)

    i = 2
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.show()

def plot_image(i, predictions_array, true_label, img):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    predictions_array, true_label, img = predictions_array, int(true_label[i]), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[int(true_label)].set_color('blue')

def main():
    #create_model()
    train()
    #import_data()

if __name__== "__main__":
  main()