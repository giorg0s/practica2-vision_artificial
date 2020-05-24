# /usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import argparse
import os

from numba.six import print_

from carga_imagenes import carga_imagenes_carpeta, CLASES
import numpy as np
from entrenamiento_lda import procesa_ocr_training
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from preprocesado import get_bounding, extraer_valores, rotar_matricula, detecta_centro

CLASIFICADOR_FRONTALES = 'assets/haar/coches.xml'
CLASIFICADOR_MATRICULAS = 'assets/haar/matriculas.xml'

# CARPETA_TEST_OCR = 'testing_ocr'
CARPETA_TRAINING_OCR = 'training_ocr'
# CARPETA_TEST_FULL_SYSTEM = 'testing_full_system'


cascade_frontales = cv2.CascadeClassifier(CLASIFICADOR_FRONTALES)
cascade_matriculas = cv2.CascadeClassifier(CLASIFICADOR_MATRICULAS)
digitos = []


def detecta_digitos(test_img):
    lista_matriculas = []
    img_trabajo = test_img.copy()
    x_mat = 0
    y_mat = 0

    x_centro = 0
    y_centro = 0

    img_color = cv2.cvtColor(img_trabajo, cv2.COLOR_GRAY2BGR)

    frontales = cascade_frontales.detectMultiScale(img_trabajo, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in frontales:
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # frontal_coche = img_trabajo[y:y + h, x:x + w]
    
    x_centro, y_centro = detecta_centro(frontales)
    cv2.circle(img_color, center= (x_centro, y_centro), radius=7, thickness=2, color=(255, 0, 0))

    matriculas = cascade_matriculas.detectMultiScale(img_trabajo, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in matriculas:
        x_mat = x
        y_mat = y
        cv2.rectangle(img_color, (x_mat, y_mat), (x_mat + w, y_mat + h), (255, 0, 255), 1)
        matricula = img_trabajo[y:y_mat + h, x:x_mat + w]  # Trozo que corresponde con la matrícula recortada
        lista_matriculas.append(matricula)

    for matricula in lista_matriculas:
        # prueba = extraer_valores(matricula)
        rotated = rotar_matricula(matricula)
        thre_mor = extraer_valores(rotated)
        # matricula_gray = cv2.bitwise_not(rotated)
        # thresh = cv2.threshold(matricula_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        ctrs, hierarchy = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lista = get_bounding(ctrs, hierarchy,  matricula.copy())

        for caracter in lista:
            x, y, w, h = caracter
            recorte = matricula[y:y + h + 3,  x:x + w + 3]
            cv2.rectangle(img_color, (x_mat + x, y_mat + y), (x_mat + x + w, y_mat + y + h), (0, 255, 0), 1)

            digito_resized = cv2.resize(recorte, (10, 10), cv2.INTER_LINEAR)
            digitos.append(digito_resized)

    if visualiza_resultados == 'True':
        cv2.imshow('RESULTADO', img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return digitos, x_centro, y_centro


def main():
    # Carga de imagenes
    test_imgs = carga_imagenes_carpeta(carpeta_img, extrae_etiquetas=False)

    # leer carpeta de imagenes -> imagenes, etiquetas
    imagenes_train, etiquetas_train = carga_imagenes_carpeta(CARPETA_TRAINING_OCR, extrae_etiquetas=True)
    caracteristicas_train, clases_train = procesa_ocr_training(imagenes_train, etiquetas_train)

    crf = LinearDiscriminantAnalysis()  # se crea el objeto de entrenador LDA
    cr7_train = crf.fit_transform(caracteristicas_train, clases_train).astype(np.float32)
    # cr7_test = crf.transform(caracteristicas_test)
    knn_clasif = KNeighborsClassifier(n_neighbors=5)  # se crea el clasificador
    knn_clasif.fit(cr7_train, clases_train)

    txt_resultados = open(carpeta_img+'.txt', 'w')

    for num_img, test_img in enumerate(test_imgs):
        print('PARA LA MATRICULA', num_img)
        digitos_leidos, x_centro, y_centro = detecta_digitos(test_img)
        mat_caracteristicas = np.zeros((len(digitos_leidos), 100), dtype=np.uint32)  # matriz de caracteristicas

        if len(digitos_leidos) > 0:
            for num_digito, digito in enumerate(digitos_leidos):
                vc = digito.flatten()
                mat_caracteristicas[num_digito, :] = vc

            cr_test = crf.transform(mat_caracteristicas)
            digitos_matricula = knn_clasif.predict(cr_test)
            digitos_leidos.clear()

            # print(list(map(lambda x: CLASES[x], digitos_matricula)))

            lista_digitos = list(map(lambda x: CLASES[x], digitos_matricula))
            string_digitos = ''.join(map(str, lista_digitos))

            resultados = str(x_centro) + ' ' + str(y_centro) + ' ' + string_digitos + '\n'
            txt_resultados.write(resultados)
        else:
            resultados = str(x_centro) + ' ' + str(y_centro) + ' ' + 'NORECONOCIDO' + '\n'
            txt_resultados.write(resultados)

    txt_resultados.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('carpeta_img', type=str, help='Carpeta de imagenes para leer')
    parser.add_argument('visualiza_resultados', type=str, help='Visualizar los resultados')
    args = parser.parse_args()

    carpeta_img = args.carpeta_img
    visualiza_resultados = args.visualiza_resultados

    main()
