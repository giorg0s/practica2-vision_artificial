# /usr/bin/python3

import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import matplotlib.pyplot as plt

from deteccion_orb import *
from deteccion_haar import *

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


def detecta_digitos(matriculas):
    char_list = []

    for i, matricula in enumerate(matriculas):
        matricula_resized = cv2.resize(matricula, (matricula.shape[1]*3, matricula.shape[0]*3))
        matricula_gray = cv2.cvtColor(matricula_resized, cv2.COLOR_BGR2GRAY)
        matricula_gray = cv2.bilateralFilter(matricula_gray, 11, 17, 17)
        # matricula_blur = cv2.GaussianBlur(matricula_gray, (7, 7), 0)

        # thresh_inv = cv2.adaptiveThreshold(matricula_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 39, 1)
        th2 = cv2.threshold(matricula_gray, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(th2, cv2.MORPH_DILATE, kernel3)

        ctrs, hierarchy = cv2.findContours(thre_mor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        lista = get_bounding(ctrs, matricula_resized)

    for rect in lista:
        x, y, w, h = rect
        caracter_pot = matricula_resized[y:y + h, x:x + w]


def procesa_ocr_training(caracteres_ocr):
    M = np.zeros((len(caracteres_ocr), 100), dtype=np.float32) # matriz de caracteristicas
    E = np.zeros((len(caracteres_ocr), 1), dtype=np.uint32)

    for i in range(len(caracteres_ocr)):
        # https://stackoverflow.com/questions/4528982/convert-alphabet-letters-to-number-in-python
        E[i][0] = int(ord(str(etiquetas[i][0])))
    num_caracter = 0

    for caracter in caracteres_ocr:
        caracter = cv2.resize(caracter, (10,10), interpolation=cv2.INTER_LINEAR)
        caracter_gray = cv2.cvtColor(caracter, cv2.COLOR_BGR2GRAY)
        caracter_blur = cv2.GaussianBlur(caracter_gray, (7, 7), 0)
        thresh_inv = cv2.adaptiveThreshold(caracter_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 39, 1)

        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(thresh_inv, cv2.MORPH_DILATE, kernel3)

        ctrs, hierarchy = cv2.findContours(thre_mor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        vc = obtener_caracteristicas(caracter_gray)
        iterador = 0
        for pixel in vc[0]:
            M[num_caracter][iterador] = vc[0][iterador]
            iterador += 1

        num_caracter += 1

    print('FIN PROCESAMIENTO')

    return (E, M)


def obtener_caracteristicas(caracter):
    # Es un vector de una sola fila y 100 columnas que contiene el valor de gris de la imagen
    vector_caracteristicas = np.zeros((1, caracter.size), dtype=np.float32)

    iterador = 0
    for fila in caracter:
        for valor_gris in fila:
            vector_caracteristicas[0][iterador] = valor_gris
            iterador += 1

    return vector_caracteristicas


def entrenamiento_lda(car, et):
    mat_proyeccion = LDA() # se crea el objeto de entrenador LDA
    mat_proyeccion.fit(car, et.ravel()) # encontrar la matriz de proyeccion

    return mat_proyeccion


def main():
    # Carga de imagenes
    test_imgs = carga_imagenes_carpeta(CARPETA_TEST, False)
    matriculas = detecta_matriculas(test_imgs)
    detecta_digitos(matriculas)

    test_ocr = carga_imagenes_carpeta(CARPETA_TRAIN_OCR, True)
    vector_etiquetas, mat_caracteristicas = procesa_ocr_training(test_ocr)
    entrenamiento_lda(mat_caracteristicas, vector_etiquetas)


if __name__ == '__main__':
    main()
