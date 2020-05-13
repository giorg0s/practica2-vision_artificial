# /usr/bin/python3
# -*- coding: utf-8 -*-

import cv2.ml
from entrenamiento_lda import *
from carga_imagenes import carga_imagenes_carpeta

CARPETA_TRAIN_OCR = 'training_ocr'


def clasificador_knn(imagen_test):
    train_ocr = carga_imagenes_carpeta(CARPETA_TRAIN_OCR, True)
    vector_etiquetas, mat_caracteristicas = procesa_ocr_training(train_ocr)
    crf, cr = reducir_dimensionalidad(mat_caracteristicas, vector_etiquetas)

    knn_clasif = cv2.ml.KNearest_create()  # se crea el clasificador
    knn_clasif.train(cr, cv2.ml.ROW_SAMPLE, vector_etiquetas) # se entrena el clasificador

    ret, result, neighbours, dist = knn_clasif.findNearest(imagen_test, k=5)  # recibe los datos de test (testing_ocr)

    print('hola')


def clasificador_bayes(mat_cr, vec_e):
    bayes_clasif = cv2.ml.NormalBayesClassifier_create()  # se crea el clasificador bayesiano
    # bayes_clasif.train(mat_cr, vec_e)

    print('LISTO')
