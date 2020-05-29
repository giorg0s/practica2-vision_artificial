# /usr/bin/python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np

RUTA_TRAIN = 'training_ocr'
TAMAÑO_TEST = 200


def procesa_ocr_training(caracteres_ocr, etiquetas):
    kernel = np.ones((2, 2), np.uint8)

    '''Preprocesar imágenes y obtener vector de características'''
    mat_caracteristicas = np.zeros((len(caracteres_ocr), 100), dtype=np.uint32)  # matriz de caracteristicas
    vec_etiquetas = np.zeros(len(etiquetas), dtype=np.uint32)

    for ix, etiqueta in enumerate(etiquetas):
        # https://stackoverflow.com/questions/4528982/convert-alphabet-letters-to-number-in-python
        vec_etiquetas[ix] = etiquetas[ix]

    for num_caracter, caracter in enumerate(caracteres_ocr):
        caracter_blur = cv2.medianBlur(caracter, 5)
        thresh = cv2.adaptiveThreshold(caracter_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        img_erode = cv2.dilate(thresh, kernel, iterations=1)

        ctrs, _ = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_ctrs = sorted(ctrs, key=lambda x: cv2.contourArea(x))

        if len(sorted_ctrs) != 0:
            x, y, w, h = cv2.boundingRect(sorted_ctrs[0])
            caracter_resized = cv2.resize(caracter[y:y + h + 2, x:x + w + 2], (10, 10), interpolation=cv2.INTER_LINEAR)

            vc = caracter_resized.flatten()

            mat_caracteristicas[num_caracter, :] = vc
        else:
            caracter_resized = cv2.resize(caracter, (10, 10), interpolation=cv2.INTER_LINEAR)
            vc = caracter_resized.flatten()

            mat_caracteristicas[num_caracter, :] = vc

    return mat_caracteristicas, vec_etiquetas


