# /usr/bin/python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from carga_imagenes import etiquetas
from preprocesado import procesa_imagen


def obtener_caracteristicas(caracter):
    # Es un vector de una sola fila y 100 columnas que contiene el valor de gris de la imagen
    vector_caracteristicas = np.zeros((1, caracter.size), dtype=np.float32)
    iterador = 0
    for fila in caracter:
        for valor_gris in fila:
            vector_caracteristicas[0][iterador] = valor_gris
            iterador += 1

    return vector_caracteristicas


def procesa_ocr_training(caracteres_ocr):
    M = np.zeros((len(caracteres_ocr), 100), dtype=np.uint32) # matriz de caracteristicas
    E = np.zeros((len(caracteres_ocr), 1), dtype=np.uint32)

    for i in range(len(caracteres_ocr)):
        # https://stackoverflow.com/questions/4528982/convert-alphabet-letters-to-number-in-python
        E[i][0] = int(ord(str(etiquetas[i][0])))

    num_caracter = 0
    for caracter in caracteres_ocr:
        caracter_resized = cv2.resize(caracter, (10,10), interpolation=cv2.INTER_LINEAR)
        thre_mor = procesa_imagen(caracter_resized)

        # ctrs, hierarchy = cv2.findContours(thre_mor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        vc = obtener_caracteristicas(thre_mor)

        iterador = 0
        for pixel in vc[0]:
            M[num_caracter][iterador] = vc[0][iterador]
            iterador += 1

        num_caracter += 1

    print('FIN PROCESAMIENTO')

    return E, M


def reducir_dimensionalidad(mat_c, vec_e):
    crf = LinearDiscriminantAnalysis()  # se crea el objeto de entrenador LDA
    crf.fit(mat_c, vec_e.ravel())  # encontrar la matriz de proyeccion

    cr = np.ndarray.astype(crf.transform(mat_c), dtype=np.float32)  # matriz de caracteristicas reducidas

    return cr  # se retorna tanto la matriz de proyeccion como la matriz CR
