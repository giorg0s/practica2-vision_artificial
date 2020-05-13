# /usr/bin/python3

import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import matplotlib.pyplot as plt

from deteccion_orb import *
from deteccion_haar import *
from preprocesado import *
from entrenamiento_lda import *
from clasificadores import clasificador_knn

CLASIFICADOR_MATRICULAS = 'assets/haar/matriculas.xml'
CARPETA_TRAIN_OCR = 'training_ocr'

cascade_matriculas = cv2.CascadeClassifier(CLASIFICADOR_MATRICULAS)


def detecta_matriculas(imagenes):
    frontales = []  # array con los frontales detectados
    matriculas = []

    for i, img in enumerate(imagenes):
        frontal_coche = procesamiento_img_haar(img)
        # frontal_coche = cv2.cvtColor(frontal_coche, cv2.COLOR_GRAY2BGR)
        frontales.append(frontal_coche)

    for n, frontal in enumerate(frontales):
        img_procesada = cascade_matriculas.detectMultiScale(frontal, scaleFactor=1.02, minNeighbors=7, minSize=(10, 10))

        if img_procesada is ():
            print('Error')
        for (x, y, w, h) in img_procesada:
            imagen_rect = cv2.rectangle(frontal, (x, y), (x + w, y + h), (0, 0, 0), 2)
            matricula = imagen_rect[y:y + h, x:x + w]
            matriculas.append(matricula)

    return matriculas


def detecta_digitos(matriculas):
    digitos = []

    for i, matricula in enumerate(matriculas):
        matricula_resized = cv2.resize(matricula, (matricula.shape[1] * 3, matricula.shape[0] * 3))
        thre_mor = procesa_imagen(matricula)
        ctrs, hierarchy = cv2.findContours(thre_mor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        lista = get_bounding(ctrs, matricula_resized)

        for rect in lista:
            x, y, w, h = rect
            recorte = matricula_resized[y:y + h, x:x + w]
            digitos.append(recorte)

    return digitos


def main():
    # Carga de imagenes
    test_imgs = carga_imagenes_carpeta(CARPETA_TEST, False)
    matriculas = detecta_matriculas(test_imgs)
    detecta_digitos(matriculas)

    test_ocr = carga_imagenes_carpeta(CARPETA_TRAIN_OCR, True)
    vector_etiquetas, mat_caracteristicas = procesa_ocr_training(test_ocr)
    crf, cr = reducir_dimensionalidad(mat_caracteristicas, vector_etiquetas)

    clasificador_knn(cr, np.ndarray.astype(vector_etiquetas, dtype=np.float32))


if __name__ == '__main__':
    main()
