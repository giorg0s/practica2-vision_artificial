# /usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from carga_imagenes import carga_imagenes_carpeta, CLASES
from entrenamiento_lda import procesa_ocr_training
from preprocesado import get_bounding, detecta_centro, procesa_imagen, rotar_matricula

import time

CLASIFICADOR_FRONTALES = 'assets/haar/coches.xml'
CLASIFICADOR_MATRICULAS = 'assets/haar/matriculas.xml'

CARPETA_TRAINING_OCR = 'training_ocr'

cascade_frontales = cv2.CascadeClassifier(CLASIFICADOR_FRONTALES)
cascade_matriculas = cv2.CascadeClassifier(CLASIFICADOR_MATRICULAS)
digitos = []


def detecta_digitos(test_img):
    img_trabajo = test_img.copy()
    img_color = cv2.cvtColor(img_trabajo, cv2.COLOR_GRAY2BGR)

    frontales = cascade_frontales.detectMultiScale(img_trabajo, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in frontales:
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if len(frontales) == 0:
        x_centro = 0
        y_centro = 0
    else:
        x_centro, y_centro = detecta_centro(frontales)
        cv2.circle(img_color, center=(x_centro, y_centro), radius=10, thickness=2, color=(255, 0, 0))

    matriculas = cascade_matriculas.detectMultiScale(img_trabajo, scaleFactor=1.02, minNeighbors=7, minSize=(10, 10),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(matriculas) > 0:
        (x, y, w, h) = matriculas[0]

        if h / w < 0.3:
            w_mat_mitad = w / 2
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (255, 0, 255), 1)
            matricula = img_trabajo[y:y + h, x:x + w]  # Trozo que corresponde con la matrícula recortada

            rotated = rotar_matricula(matricula.copy())
            thre_mor = procesa_imagen(rotated)
            ctrs, hierarchy = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            lista = get_bounding(ctrs, matricula.copy())

            for digito in lista:
                x_digito, y_digito, w_digito, h_digito = digito
                recorte = matricula[y_digito:y_digito + h_digito + 3, x_digito:x_digito + w_digito + 3]
                cv2.rectangle(img_color, (x + x_digito, y + y_digito),
                              (x + x_digito + w_digito, y + y_digito + h_digito), (0, 255, 0), 1)

                digito_resized = cv2.resize(recorte, (10, 10), cv2.INTER_LINEAR)
                digitos.append(digito_resized)

            visualizar_resultados(img_color)
            return digitos, x_centro, y_centro, w_mat_mitad

    else:
        visualizar_resultados(img_color)
        return [], 0, 0, 0.0


def visualizar_resultados(imagen):
    if visualiza_resultados == 'True':
        cv2.imshow('RESULTADO', imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    inicio = time.time()
    # leer carpeta de imagenes -> imagenes, etiquetas
    imagenes_train, etiquetas_train = carga_imagenes_carpeta(CARPETA_TRAINING_OCR, extrae_etiquetas=True)
    caracteristicas_train, clases_train = procesa_ocr_training(imagenes_train, etiquetas_train)

    crf = LinearDiscriminantAnalysis()  # se crea el objeto de entrenador LDA
    cr7_train = crf.fit_transform(caracteristicas_train, clases_train).astype(np.float32)
    # cr7_test = crf.transform(caracteristicas_test)
    knn_clasif = KNeighborsClassifier(n_neighbors=5)  # se crea el clasificador
    knn_clasif.fit(cr7_train, clases_train)

    txt_resultados = open(carpeta_img + '.txt', 'w')

    for img in glob.glob(carpeta_img + '/' + '*.jpg'):
        imagen = cv2.imread(img, 0)
        digitos_leidos, x_centro, y_centro, w_mat = detecta_digitos(imagen)

        if len(digitos_leidos) > 0:
            mat_caracteristicas = np.zeros((len(digitos_leidos), 100), dtype=np.uint32)  # matriz de caracteristicas
            for num_digito, digito in enumerate(digitos_leidos):
                vc = digito.flatten()
                mat_caracteristicas[num_digito, :] = vc

            cr_test = crf.transform(mat_caracteristicas)
            digitos_matricula = knn_clasif.predict(cr_test)
            digitos_leidos.clear()

            # print(list(map(lambda x: CLASES[x], digitos_matricula)))

            lista_digitos = list(map(lambda x: CLASES[x], digitos_matricula))
            string_digitos = ''.join(map(str, lista_digitos))

            resultados = os.path.basename(img) + ' ' + str(x_centro) + ' ' + str(
                y_centro) + ' ' + string_digitos + ' ' + str(w_mat) + '\n'
            txt_resultados.write(resultados)
        else:
            resultados = os.path.basename(img) + ' ' + str(x_centro) + ' ' + str(
                y_centro) + ' ' + 'NORECONOCIDO' + ' ' + str(w_mat) + '\n'
            txt_resultados.write(resultados)

    txt_resultados.close()
    fin = time.time()

    print('TIEMPO TOTAL:', fin-inicio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('carpeta_img', type=str, help='Carpeta de imagenes para leer')
    parser.add_argument('visualiza_resultados', type=str, help='Visualizar los resultados')
    args = parser.parse_args()

    carpeta_img = args.carpeta_img
    visualiza_resultados = args.visualiza_resultados

    main()
