# /usr/bin/python3
# -*- coding: utf-8 -*-
from builtins import print

import cv2


def procesa_imagen(imagen):
    # imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_gray = cv2.bilateralFilter(imagen, 11, 17, 17)
    # matricula_blur = cv2.GaussianBlur(matricula_gray, (7, 7), 0)

    # thresh_inv = cv2.adaptiveThreshold(matricula_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
    # 39, 1)
    th2 = cv2.threshold(imagen_gray, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(th2, cv2.MORPH_DILATE, kernel3)

    return thre_mor


def get_bounding(ctrs, imagen):
    char_list = []
    digitos = []
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        if (w < imagen.shape[1]*0.12) and (w > imagen.shape[1]*0.02) and (h < imagen.shape[0]*0.6) and (h > imagen.shape[0]*0.45):
            # cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 0, 0), 2)
            x_resized = int(x/5)
            y_resized = int(y/5)
            w_resized = int(w/5)
            h_resized = int(h/5)
            char_list.append((x_resized, y_resized, w_resized, h_resized))
            char_list.sort()

    for rect in char_list:
        x, y, w, h = rect
        digitos.append((x, y, w, h))

    return digitos
