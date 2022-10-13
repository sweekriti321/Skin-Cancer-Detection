from keras.models import load_model
import cv2
import numpy as np

model = load_model('cancer_model.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['acc'])

img = cv2.imread('./data/image.jpg')
img = cv2.resize(img,(32,32))
img = np.reshape(img,[None,32,32,3])

# pred_probab = model.predict(img)
pred_probab, pred_class = keras_predict(model, data)
print (pred_probab)

# classes = model.predict_classes(img)
#
# print (classes)