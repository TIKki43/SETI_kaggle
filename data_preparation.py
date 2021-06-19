from keras_preprocessing.image import ImageDataGenerator
import skimage.io as io
from PIL import Image
import pandas as pd
import numpy as np
import shutil
import cv2
import os

tst_pos, tst_ngtv = 123, 123
# tst_pos, tst_ngtv = 0, 0
data = pd.read_csv('train_labels.csv')
for i in os.listdir('/home/timur/Documents/Projects/SETI/data/train/1/'):
    if i[:-4] in data[data.columns[0]].to_list():
        idx = data[data.columns[0]].to_list().index(i[:-4])
        # print(data[data.columns[1]].to_list()[idx])
        # break
        if data[data.columns[1]].to_list()[idx] == 1 and tst_pos <= 10:
            tst_pos += 1
            shutil.move(f'/home/timur/Documents/Projects/SETI/data/train/1/{i}',
                        f'/home/timur/Documents/Projects/SETI/data/test/{i}')
        elif data[data.columns[1]].to_list()[idx] == 0 and tst_ngtv <= 10:
            tst_ngtv += 1
            shutil.move(f'/home/timur/Documents/Projects/SETI/data/train/1/{i}',
                        f'/home/timur/Documents/Projects/SETI/data/test/{i}')
        elif data[data.columns[1]].to_list()[idx] == 0 and tst_ngtv > 10:
            shutil.move(f'/home/timur/Documents/Projects/SETI/data/train/1/{i}',
                        f'/home/timur/Documents/Projects/SETI/data/train/0/{i}')
        else:
            shutil.move(f'/home/timur/Documents/Projects/SETI/data/train/1/{i}',
                        f'/home/timur/Documents/Projects/SETI/data/train/1/{i}')

print(len(os.listdir('/home/timur/Documents/Projects/SETI/data/train/1/')))
print(len(os.listdir('/home/timur/Documents/Projects/SETI/data/train/0/')))
print(len(os.listdir('/home/timur/Documents/Projects/SETI/data/test/')))

# path_to_dir = '/home/timur/Documents/Projects/SETI/data/test/'
# g = -1
# for i in os.listdir(path_to_dir):
#     g += 1
#     os.renames(path_to_dir + f'{i}', path_to_dir + f'{g}'+'.png')

def testGenerator(test_path, num_image, target_size):
    for i in range(num_image):
        img = cv2.imread(os.path.join(test_path, "%d.png" % i)).astype(float)
        img = img / 255
        img = np.resize(img, target_size)
        img = np.reshape(img, (-1, 150, 150, 3))
        yield img


train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
        '/home/timur/Documents/Projects/SETI/data/train/',
        target_size=(150, 150),
        batch_size=10)

test_generator = testGenerator('/home/timur/Documents/Projects/SETI/data/test/', 3, (150, 150, 3))

# print(len(train_generator))