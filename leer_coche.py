# /usr/bin/python3

import cv2
import sklearn
import numpy as np
from deteccion_orb import *
from deteccion_haar import *

CLASIFICADOR_MATRICULAS = 'assets/haar/matriculas.xml'

cascade_matriculas = cv2.CascadeClassifier(CLASIFICADOR_MATRICULAS)


def detecta_matriculas(imagenes):
    frontales = [] # array con los frontales detectados
    for i, img in enumerate(imagenes):
        frontal_coche = procesamiento_img_haar(img)
        # frontal_coche = cv2.cvtColor(frontal_coche, cv2.COLOR_GRAY2BGR)
        frontales.append(frontal_coche)

    for i, frontal in enumerate(frontales):
        img_procesada = cascade_matriculas.detectMultiScale(frontal, scaleFactor=1.02, minNeighbors=7, minSize=(10, 10))

        if img_procesada is ():
            print('Error')
        for (x, y, w, h) in img_procesada:
            imagen_rect = cv2.rectangle(frontal, (x, y), (x + w, y + h), (255, 0, 0), 2)
            matricula = imagen_rect[y:y+h, x:x+w]
            matricula_gray = cv2.cvtColor(matricula, cv2.COLOR_BGR2GRAY)
            matricula_blur = cv2.medianBlur(matricula_gray, 3)

            th = cv2.adaptiveThreshold(matricula_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                       cv2.THRESH_BINARY, 11, 2)
            contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for ctr in contours:
                x, y, w, h = cv2.boundingRect(ctr)

            cv2.drawContours(matricula, contours, -1, (0,255,0), 1)

            cv2.imshow('Contours', frontal)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # cv2.imshow("Detector de matriculas", imagen_rect)
            # cv2.waitKey(0)
            # time.sleep(1)
            # return imagen_rect


def main():
    # Carga de imagenes
    test_imgs = carga_imagenes_carpeta(CARPETA_TEST)
    detecta_matriculas(test_imgs)


if __name__ == '__main__':
    main()