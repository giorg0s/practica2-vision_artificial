# /usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import argparse
from carga_imagenes import carga_imagenes_carpeta, CLASES
import numpy as np
from entrenamiento_lda import procesa_ocr_training
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from preprocesado import procesa_imagen, get_bounding, extraer_valores, rotar_matricula

CLASIFICADOR_FRONTALES = 'assets/haar/coches.xml'
CLASIFICADOR_MATRICULAS = 'assets/haar/matriculas.xml'

# CARPETA_TEST_OCR = 'testing_ocr'
CARPETA_TRAINING_OCR = 'training_ocr'
# CARPETA_TEST_FULL_SYSTEM = 'testing_full_system'

FICHERO_SALIDA = 'resultados.txt'

cascade_frontales = cv2.CascadeClassifier(CLASIFICADOR_FRONTALES)
cascade_matriculas = cv2.CascadeClassifier(CLASIFICADOR_MATRICULAS)
digitos = []


def detecta_digitos(test_img):
    lista_matriculas = []
    img_trabajo = test_img.copy()
    x_mat = 0
    y_mat = 0

    img_color = cv2.cvtColor(img_trabajo, cv2.COLOR_GRAY2BGR)

    frontales = cascade_frontales.detectMultiScale(img_trabajo, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in frontales:
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
        frontal_coche = img_trabajo[y:y + h, x:x + w]

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
            recorte = matricula[y:y + h + 2,  x:x + w + 2]
            cv2.rectangle(img_color, (x_mat + x, y_mat + y), (x_mat + x + w, y_mat + y + h), (0, 255, 0), 1)

            digito_resized = cv2.resize(recorte, (10, 10), cv2.INTER_LINEAR)
            digitos.append(digito_resized)

    if visualiza_resultados == 'True':
        cv2.imshow('RESULTADO', img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return digitos


def main():
    # Carga de imagenes
    test_imgs = carga_imagenes_carpeta(carpeta_img, False)

    # leer carpeta de imagenes -> imagenes, etiquetas
    imagenes_train, etiquetas_train = carga_imagenes_carpeta(CARPETA_TRAINING_OCR, extrae_etiquetas=True)
    caracteristicas_train, clases_train = procesa_ocr_training(imagenes_train, etiquetas_train)

    crf = LinearDiscriminantAnalysis()  # se crea el objeto de entrenador LDA
    cr7_train = crf.fit_transform(caracteristicas_train, clases_train).astype(np.float32)
    # cr7_test = crf.transform(caracteristicas_test)
    knn_clasif = KNeighborsClassifier(n_neighbors=5)  # se crea el clasificador
    knn_clasif.fit(cr7_train, clases_train)

    for num_img, test_img in enumerate(test_imgs):
        print('PARA LA MATRICULA', num_img)
        digitos_leidos = detecta_digitos(test_img)
        mat_caracteristicas = np.zeros((len(digitos_leidos), 100), dtype=np.uint32)  # matriz de caracteristicas

        if len(digitos_leidos) > 0:
            for num_digito, digito in enumerate(digitos_leidos):
                # digito_proc = procesa_imagen(digito)
                vc = digito.flatten()
                mat_caracteristicas[num_digito, :] = vc

            cr_test = crf.transform(mat_caracteristicas)
            digitos_matricula = knn_clasif.predict(cr_test)
            digitos_leidos.clear()

            print(list(map(lambda x: CLASES[x], digitos_matricula)))
        else:
            print('NO RECONOCIDO')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('carpeta_img', type=str, help='Carpeta de imagenes para leer')
    parser.add_argument('visualiza_resultados', type=str, help='Visualizar los resultados')
    args = parser.parse_args()

    carpeta_img = args.carpeta_img
    visualiza_resultados = args.visualiza_resultados

    main()
