# /usr/bin/python3

import cv2
import time
import numpy as np

CARPETA_TEST = 'img/test'
CLASIFICADOR = 'assets/haar/coches.xml'

# Se crea el cascade classifier
cascade = cv2.CascadeClassifier(CLASIFICADOR)


def procesamiento_img_haar(imagen):
    # Se convierte la imagen a escala de grises
    # gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # El código se basa en los ejemplos que se encuentran en la web: https://www.programcreek.com/python/example/79435/cv2.CascadeClassifier
    imagen_eq = cv2.equalizeHist(imagen)

    frontales = cascade.detectMultiScale(imagen_eq, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

    if imagen_gris is ():
        print('ERROR')


        return frontal_detectado

    # cv2.imwrite(os.path.join('img', 'output', 'output_haar'+str(contador)+'.png'), imagen)

    # cv2.destroyAllWindows()
