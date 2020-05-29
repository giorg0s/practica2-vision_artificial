# /usr/bin/python3
# -*- coding: utf-8 -*-
from builtins import print

import cv2
import numpy as np
import math

kernel = np.ones((5, 5), np.uint8)


def detecta_centro(frontales):
    if (len(frontales)) > 0:
        x, y, w, h = frontales[0]
        centro_x = int((x + w) / 2)
        centro_y = int((y + h) / 2)

        return centro_x, centro_y
    else:
        print('NO SE HA ENCONTRADO EL CENTRO')


def procesa_imagen(imagen):
    imagen_blur = cv2.medianBlur(imagen, 5)
    th2 = cv2.adaptiveThreshold(imagen_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)

    return th2


def get_bounding(ctrs, imagen):
    char_list = []
    digitos = []
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        largo_matricula = imagen.shape[0]
        ancho_matricula = imagen.shape[1]

        if math.floor(largo_matricula*0.40) < h and math.ceil(ancho_matricula * 0.02) < w < math.ceil(ancho_matricula * 0.18):
            char_list.append((x, y, w, h))
            char_list.sort()

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
