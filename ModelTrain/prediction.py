import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import cv2
import os
from PIL import Image
from numpy import asarray
from tensorflow import keras
import keras
import sqlite3
import numpy as np
from keras.models import load_model

from keras.preprocessing.image import image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt
import cv2

# print (tf.__version__)

img_path = "./data/image4.jpg"
img = Image.open("./data/image4.jpg")
# plt.imshow(img)
# plt.show()

img = image.load_img(img_path,target_size=(64,64))

# test_image = image.load_image('data/image.jpg',target_size = (32,32))
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)
# result = cnn.predict(img)

# print (result)
# img = img.resize((32, 32))
# data = asarray(img)
# data = np.reshape(img,[None,32,32,3])
# plt.imshow(data)
# plt.show()
#
# data = asarray(img)
# print(data)

#img_array=image.img_to_array(img)
#img_batch=np.expand_dims(img_array,axis=0)

model = load_model('./cancer_model_main.h5')
pred_probab = model.predict(img)

# print(decode_predictions(pred_probab, top=8 )[0][0])
#  print(pred_probab)

 # print("After prediction our output is : ")
 #    print (pred_probab)
maxElement = np.amax(pred_probab)

print('Max   : ', maxElement)
max_index_row = np.argmax(pred_probab)
# 1 finds row indices

if(max_index_row == 0):
    print("actinic keratosis(akiec)")
elif(max_index_row == 1):
    print("basal cell carcinoma(bcc)")
elif (max_index_row == 2):
    print("Benign or Non-Cancerous(ben)")
elif(max_index_row == 3):
    print("benign keratosis(bkl)")
elif(max_index_row == 4):
    print("dermatofibroma")
elif(max_index_row == 5):
    print("melanoma")
elif(max_index_row == 6):
    print("melanocytic nevus")
elif(max_index_row == 7):
    print("vascular lesion")
else:
    print("Prediction probability less than 70 percent")


print(max_index_row)

# result = np.where(pred_probab == np.amax(pred_probab))
# print('Returned tuple of arrays :', result)
# print('List of Indices of maximum element :', result)
# except Exception as e :
#    print ("Some Exception occurs")
# Find the index of the max value



# Return the max value of the list

#image_preprocessed = preprocess_input(img_batch)
#model=tf.keras.applications.resnet50.ResNet50('cancer_model.h5')

#prediction = model.predict(image_preprocessed)
#print (decode_predictions(prediction))
#   5    6705 nv
#   4    1113 mel
#   2    1099 bkl
#   1     514 bcc
#   0     327 akiec
#   6     142 vasc
#   3     115 df melanoma, melanocytic nevus, basal cell carcinoma, actinic keratosis, benign keratosis, dermatofibroma, and vascular lesion.