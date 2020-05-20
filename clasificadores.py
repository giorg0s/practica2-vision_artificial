# /usr/bin/python3
# -*- coding: utf-8 -*-

import cv2.ml
from entrenamiento_lda import *
from carga_imagenes import carga_imagenes_carpeta
import matplotlib.pyplot as plt

CARPETA_TRAIN_OCR = 'training_ocr'

knn_clasif = cv2.ml.KNearest_create()  # se crea el clasificador


def preparar_clasificador_knn():
    train_ocr = carga_imagenes_carpeta(CARPETA_TRAIN_OCR, True)
    vector_etiquetas, mat_caracteristicas = procesa_ocr_training(train_ocr)
    crf, cr = reducir_dimensionalidad(mat_caracteristicas, vector_etiquetas)

    knn_clasif.train(cr, cv2.ml.ROW_SAMPLE, vector_etiquetas.astype(np.float32))  # se entrena el clasificador

    return knn_clasif  # se retorna el clasificador entrenado


def aplicar_clasfificador_knn(imagen_test, clasificador_knn):
    ret, results, neighbours, dist = clasificador_knn.findNearest(imagen_test, k=5)  # recibe los datos de test (testing_ocr)

    print("result:  {}\n".format(results))
    print("neighbours:  {}\n".format(neighbours))
    print("distance:  {}\n".format(dist))

    plt.show()


def clasificador_bayes(mat_cr, vec_e):
    bayes_clasif = cv2.NormalBayesClassifier()  # se crea el clasificador bayesiano
    # bayes_clasif.train(mat_cr, vec_e)

    print('LISTO')
