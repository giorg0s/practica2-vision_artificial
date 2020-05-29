# /usr/bin/python3
# -*- coding: utf-8 -*-

import glob
import os
import cv2
import numpy as np

CLASES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
          'B', 'C', 'D', 'E', 'ESP', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
          'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


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
            if os.path.basename(img)[0:3] == 'ESP':
                charname = 'ESP'
            else:
                charname = os.path.basename(img)[0]
            etiqueta = CLASES.index(charname)
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
