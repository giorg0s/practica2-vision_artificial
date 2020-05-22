# /usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import os
import glob
import time
import numpy as np


def carga_imagenes_carpeta(nombre_carpeta, extrae_etiquetas):
    imagenes = []
    etiquetas = []
    print("Se va a iniciar la carga de las imagenes de", nombre_carpeta)
    print("###################################################")
    # time.sleep(2)

    for img in glob.glob(nombre_carpeta + '/' + '*.jpg'):
        imagen = cv2.imread(img, 0)
        imagenes.append(imagen)
        if extrae_etiquetas:
            charname = os.path.basename(img)[0]
            etiqueta = int(ord(charname))
            etiquetas.append(etiqueta)
        # print("He leido la imagen ", img)
        # time.sleep(.100)

    print("###################################################")
    print("FIN")
    print()
    if extrae_etiquetas:
        return np.array(imagenes), np.array(etiquetas)
    else:
        return np.array(imagenes)
