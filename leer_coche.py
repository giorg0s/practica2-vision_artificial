# /usr/bin/python3

import cv2
import sklearn
import numpy as np
from deteccion_orb import *
from deteccion_haar import *

CLASIFICADOR_MATRICULAS = 'assets/haar/matriculas.xml'

cascade_matriculas = cv2.CascadeClassifier(CLASIFICADOR_MATRICULAS)

def detecta_matriculas(imagenes):
    # frontales = [] # array con los frontales detectados
    for i, img in enumerate(imagenes):
        frontal_coche = procesamiento_img_haar(img, i)
        img_procesada = cascade_matriculas.detectMultiScale(frontal_coche, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

        if img_procesada is ():
            print('Error')
        for (x, y, w, h) in img_procesada:
            imagen_rect = cv2.rectangle(frontal_coche, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('Detector de coches', imagen_rect[y:y + h, x:x + w])

            cv2.waitKey(1)
            # time.sleep(1)


def main():
    # Carga de imagenes
    test_imgs = carga_imagenes_carpeta(CARPETA_TEST)
    detecta_matriculas(test_imgs)


if __name__ == '__main__':
    main()