import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import cv2

test_image_dir = 'Write your image path'
test_list_img = glob.glob(test_image_dir + '/*.*')

def plt_display(image, title):
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(image)
    a.set_title(title)

img_cv = []
img_tf = []
for i in range(len(test_list_img)):
    img_cv = cv2.imread(test_list_img[i])
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)/255
    img_cv = cv2.resize(img_cv,(224,224),interpolation=cv2.INTER_LINEAR)
    
    img_tf = tf.io.read_file(test_list_img[i])
    img_tf = tf.image.decode_jpeg(img_tf, channels=3, dct_method="INTEGER_ACCURATE")/255
    img_tf = tf.image.resize(img_tf, [224, 224])
    
    img_diff = np.abs(img_cv - img_tf.numpy())
    plt_display(img_diff, 'TENSORFLOW & OPENCV IS CHECKING...')
