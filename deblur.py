import cv2
import numpy as np

def deblur(img_path, output_path):
    image = cv2.imread(img_path)
    kernel = np.ones((5, 5), np.float32) / 25
    deblurred = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(output_path, deblurred)

deblur("d:/Imaging/blurred.png", "/deblurred.png")