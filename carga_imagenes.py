# /usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import os
import glob
import time

etiquetas = []


def carga_imagenes_carpeta(nombre_carpeta, extrae_etiquetas):
    imagenes = []
    print("Se va a iniciar la carga de las imagenes de", nombre_carpeta)
    print("###################################################")
    # time.sleep(2)

    for img in glob.glob(nombre_carpeta + '/' + '*.jpg'):
        imagen = cv2.imread(img, 0)
        imagenes.append(imagen)
        if extrae_etiquetas:
            etiquetas.append(os.path.basename(img)[0])
        # print("He leido la imagen ", img)
        # time.sleep(.100)

    print("###################################################")
    print("FIN")
    print()

    return imagenes
