import numpy as np
import os
import cv2

IMAGE_DIR_PATH = '/home/jessica/Downloads/data_road/training/gt_image_2'
image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]


def getLabels(imgs, filename):
    labels = []
    print(imgs.shape)
    img = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            #print(imgs[i][j][k].numpy())
            if (imgs[i][j] == [1,0,1]).all():
                row.append(1)
            else:
                row.append(0)
        img.append(row)
        np.save('/home/jessica/Downloads/data_road/labels/' + filename, img)

def checkLabel(filename):
    labels = np.load(filename, allow_pickle=True)
    print(labels)

for image_path in image_paths:
    img = cv2.imread(image_path) / 255
    filename = image_path.split("/")[-1].strip('.png')
    print(filename)
    getLabels(img, filename)

checkLabel('/home/jessica/Downloads/data_road/labels/um_road_000005.npy')
