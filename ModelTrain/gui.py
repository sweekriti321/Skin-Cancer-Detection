import sys
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QLabel, QFileDialog
from PyQt5.uic import loadUi
# from rec_web import *
import numpy as np
import cv2
import pickle
import numpy as np
# import tensorflow as tf
# from cnn_tf import cnn_model_fn
import os
import sqlite3
from keras.models import load_model


class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.imagePath = None
        self.fname = None
        loadUi('gui_file.ui', self)

        self.font = QFont()
        self.font.setFamily("Arial")
        self.font.setPointSize(35)
        self.output.setFont(self.font)
        self.upload_button.clicked.connect(self.image_upload)

    def image_upload(self):
        new_path = "./data/image5.jpg"
        self.fname = QFileDialog.getOpenFileName(self, 'Open file', 'C:/', "Image files (*.jpg *.gif *.png)")
        print ('dx')
        self.imagePath = self.fname[0]
        print('cx')
        pix= QPixmap("C:/Users/Acer/PycharmProjects/ModelTrain/data/reorganized/bkl/ISIC_0024324.jpg")
        print('ex')
        label_image = QLabel()
        label_image.setPixmap(pix)
        # cv2.imshow(QPixmap(pixmap))
        # print('fx')
        # # quit()
        # cv2.waitKey(50000)
        # print('gx')
        # cv2.imwrite(new_path, pixmap)
        # print('hx')
        # self.output_image.setPixmap(QPixmap(pixmap))
        # print('ix')
        # self.resize(64, 64)
        # print('x')


    # def update_frame(self):
    #     text = " "
    #     ret, self.image = self.capture.read()
    #     self.image = cv2.flip(self.image, 1)
    #     self.image = cv2.resize(self.image, (640, 480))
    #     cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     self.imgCrop = self.image[y:y + h, x:x + w]
    #     self.imgHSV = cv2.cvtColor(self.imgCrop, cv2.COLOR_BGR2HSV)
    #     hist = get_hand_hist()
    #     dst = cv2.calcBackProject([self.imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    #     disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    #     cv2.filter2D(dst, -1, disc, dst)
    #     blur = cv2.GaussianBlur(dst, (11, 11), 0)
    #     blur = cv2.medianBlur(blur, 15)
    #     thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #     thresh = cv2.merge((thresh, thresh, thresh))
    #     thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    #     self.thresh = thresh[y:y + h, x:x + w]
    #
    #     contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    #     if len(contours) > 0:
    #         contour = max(contours, key=cv2.contourArea)
    #         # print(cv2.contourArea(contour))
    #         if cv2.contourArea(contour) > 10000:
    #             x1, y1, w1, h1 = cv2.boundingRect(contour)
    #             save_img = thresh[y1:y1 + h1, x1:x1 + w1]
    #
    #             if w1 > h1:
    #                 save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
    #                                               cv2.BORDER_CONSTANT, (0, 0, 0))
    #             elif h1 > w1:
    #                 save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
    #                                               cv2.BORDER_CONSTANT, (0, 0, 0))
    #
    #             pred_probab, pred_class = keras_predict(model, save_img)
    #             if pred_probab * 100 > 80:
    #                 text = " "
    #                 text = get_pred_text_from_db(pred_class)
    #                 print(text)
    #                 self.myText.setText(text)
    #
    #     self.displayImage(self.image)
    #     self.displayThresh(thresh)
    #
    # def displayImage(self, img):
    #     qformat = QImage.Format_Indexed8
    #     if len(img.shape) == 3:  # [0]=rows, [1]=cols, [2]=channels
    #         if img.shape[2] == 4:
    #             qformat = QImage.Format_RGBA8888
    #         else:
    #             qformat = QImage.Format_RGB888
    #
    #     outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
    #     # BGR to RGB
    #     outImage = outImage.rgbSwapped()
    #     self.camera_display.setPixmap(QPixmap.fromImage(outImage))
    #     self.camera_display.setScaledContents(True)
    #
    # def displayThresh(self, img):
    #     qformat = QImage.Format_Indexed8
    #     if len(img.shape) == 3:  # [0]=rows, [1]=cols, [2]=channels
    #         if img.shape[2] == 4:
    #             qformat = QImage.Format_RGBA8888
    #         else:
    #             qformat = QImage.Format_RGB888
    #
    #     outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
    #     # BGR to RGB
    #     outImage = outImage.rgbSwapped()
    #     self.threshhold_display.setPixmap(QPixmap.fromImage(outImage))
    #     self.threshhold_display.setScaledContents(True)
    #
    # def stop_webcam(self):
    #     self.capture.release()
    #     self.timer.stop()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('Skin Cancer Detection')
    window.show()
    sys.exit(app.exec_())