# /usr/bin/python3
# -*- coding: utf-8 -*-
from builtins import print

import cv2
import numpy as np


def detecta_centro(frontales):
    if(len(frontales)) > 0:
        x, y, w, h = frontales[0]
        centro_x = int((x + w)/2)
        centro_y = int((y + h)/2)

        return centro_x, centro_y
    else:
        print('NO SE HA ENCONTRADO EL CENTRO')


def extraer_valores(imagen):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    V = cv2.split(cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV))[2]
    T = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 75, 10)
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)

    return thresh


def procesa_imagen(imagen):
    imagen_gray = cv2.bilateralFilter(imagen, 11, 17, 17)
    th2 = cv2.adaptiveThreshold(imagen_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 75, 10)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(th2, cv2.MORPH_DILATE, kernel3)

    return thre_mor


def get_bounding(ctrs, hierarchy,  imagen):
    char_list = []
    digitos = []
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    ratio = imagen.shape[1]/imagen.shape[0]

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        ratio_contorno = h/float(w)
        if ratio_contorno > 1 and h > 10 and w >= 6:
            char_list.append((x, y, w, h))
            char_list.sort()
        # if (w < imagen.shape[1] * 0.17) and (w > imagen.shape[1] * 0.02) and (h < imagen.shape[0] * 0.95) and (
        #         h > imagen.shape[0] * 0.4):
        #     char_list.append((x, y, w, h))
        #     char_list.sort()
    for rect in char_list:
        x, y, w, h = rect
        digitos.append((x, y, w, h))

    return digitos


# BASADO EN:
# https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
def rotar_matricula(matricula):
    matricula_gray = cv2.bitwise_not(matricula)
    thresh = cv2.threshold(matricula_gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = matricula.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(matricula, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated
