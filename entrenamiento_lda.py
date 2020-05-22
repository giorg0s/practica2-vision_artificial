# /usr/bin/python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from carga_imagenes import carga_imagenes_carpeta
from preprocesado import procesa_imagen
from random import shuffle


RUTA_TRAIN = 'training_ocr'
TAMAÑO_TEST = 200

# def obtener_caracteristicas(caracter):
#     # Es un vector de una sola fila y 100 columnas que contiene el valor de gris de la imagen
#     vector_caracteristicas = np.zeros((1, 100), dtype=np.float32)
#     iterador = 0
#     for fila in caracter:
#         for valor_gris in fila:
#             vector_caracteristicas[0][iterador] = valor_gris
#             iterador += 1

#     return vector_caracteristicas


def procesa_ocr_training(caracteres_ocr, etiquetas):
    '''Preprocesar imágenes y obtener vector de características'''
    mat_caracteristicas = np.zeros((len(caracteres_ocr), 100), dtype=np.uint32)  # matriz de caracteristicas
    vec_etiquetas = np.zeros(len(etiquetas), dtype=np.uint32)

    for ix, etiqueta in enumerate(etiquetas):
        # https://stackoverflow.com/questions/4528982/convert-alphabet-letters-to-number-in-python
        vec_etiquetas[ix] = etiquetas[ix]

    for num_caracter, caracter in enumerate(caracteres_ocr):
        caracter_resized = cv2.resize(caracter, (10, 10), interpolation=cv2.INTER_LINEAR)
        thre_mor = procesa_imagen(caracter_resized)

        # ctrs, hierarchy = cv2.findContours(thre_mor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        vc = thre_mor.flatten()

        mat_caracteristicas[num_caracter, :] = vc

    print('FIN pero sin fliparse demasiado tampoco')

    return mat_caracteristicas, vec_etiquetas


# def reducir_dimensionalidad(mat_c, vec_e):
#     crf = LinearDiscriminantAnalysis()  # se crea el objeto de entrenador LDA
#     crf.fit(mat_c, vec_e)  # encontrar la matriz de proyeccion

#     cr = np.ndarray.astype(crf.transform(mat_c), dtype=np.float32)  # matriz de caracteristicas reducidas

#     return cr  # se retorna tanto la matriz de proyeccion como la matriz CR

def clasificador(caracteristicas_test):
    # leer carpeta de imagenes -> imagenes, etiquetas
    imagenes_train, etiquetas_train = carga_imagenes_carpeta(RUTA_TRAIN, extrae_etiquetas=True)

    # random_ix = np.random.choice(len(imagenes), len(imagenes))
    # imagenes_train, etiquetas_train = imagenes[random_ix[:-TAMAÑO_TEST]], etiquetas[random_ix[:-TAMAÑO_TEST]]
    # imagenes_test, etiquetas_test = imagenes[random_ix[TAMAÑO_TEST:]], etiquetas[random_ix[TAMAÑO_TEST:]]

    # obtener caracteristicas
    caracteristicas_train, clases_train = procesa_ocr_training(imagenes_train, etiquetas_train)
    # caracteristicas_test, clases_test = procesa_ocr_training(imagenes_test, etiquetas_test)
    # entrenar un lda -> Lda
    crf = LinearDiscriminantAnalysis()  # se crea el objeto de entrenador LDA
    cr7_train = crf.fit_transform(caracteristicas_train, clases_train).astype(np.float32)  
    cr7_test = crf.transform(caracteristicas_test)
    # predicciones_test = crf.predict(caracteristicas_test)

    knn_clasif = KNeighborsClassifier(n_neighbors=1)  # se crea el clasificador
    knn_clasif.fit(cr7_train, clases_train)
    predicciones_test = knn_clasif.predict(cr7_test)

    print('mudito')
    # print(np.mean(clases_test == predicciones_test))

