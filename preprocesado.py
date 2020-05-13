# /usr/bin/python3
# -*- coding: utf-8 -*-

import cv2


def procesa_imagen(imagen):
    imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen_gray = cv2.bilateralFilter(imagen_gray, 11, 17, 17)
    # matricula_blur = cv2.GaussianBlur(matricula_gray, (7, 7), 0)

    # thresh_inv = cv2.adaptiveThreshold(matricula_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
    # 39, 1)
    th2 = cv2.threshold(imagen_gray, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(th2, cv2.MORPH_DILATE, kernel3)

    return thre_mor


def get_bounding(ctrs, imagen):
    char_list = []
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        if (w < imagen.shape[1]*0.3) and (w > imagen.shape[1]*0.02) and (h > imagen.shape[0]*0.45):
            char_list.append((x, y, w, h))
            char_list.sort()

    return char_list
