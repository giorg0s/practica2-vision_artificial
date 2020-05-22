# /usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
from carga_imagenes import carga_imagenes_carpeta, CLASES
import numpy as np
from entrenamiento_lda import procesa_ocr_training
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from preprocesado import procesa_imagen
from preprocesado import get_bounding


CLASIFICADOR_FRONTALES = 'assets/haar/coches.xml'
CLASIFICADOR_MATRICULAS = 'assets/haar/matriculas.xml'
CARPETA_TEST_OCR = 'testing_ocr'
CARPETA_TRAINING_OCR = 'training_ocr'
CARPETA_TEST_FULL_SYSTEM = 'testing_full_system'

cascade_frontales = cv2.CascadeClassifier(CLASIFICADOR_FRONTALES)
cascade_matriculas = cv2.CascadeClassifier(CLASIFICADOR_MATRICULAS)
digitos = []


def detecta_digitos(test_img):
    lista_matriculas = []
    lista = []
    img_trabajo = test_img.copy()
    x_mat = 0
    y_mat = 0

    img_color = cv2.cvtColor(img_trabajo, cv2.COLOR_GRAY2BGR)

    frontales = cascade_frontales.detectMultiScale(img_trabajo, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in frontales:
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
        frontal_coche = img_trabajo[y:y + h, x:x + w]

    matriculas = cascade_matriculas.detectMultiScale(img_trabajo, scaleFactor=1.2, minNeighbors=4, minSize=(10, 10),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in matriculas:
        x_mat = x
        y_mat = y
        cv2.rectangle(img_color, (x_mat, y_mat), (x_mat+w, y_mat+h), (255, 0, 255), 1)
        matricula = img_trabajo[y:y_mat + h, x:x_mat + w]  # Trozo que corresponde con la matrícula recortada
        lista_matriculas.append(matricula)

    for matricula in lista_matriculas:
        matricula_resized = matricula.copy()
        matricula_resized = cv2.resize(matricula_resized, (matricula.shape[1] * 5, matricula.shape[0] * 5), cv2.INTER_LINEAR)

        # cv2.imshow('MATRICULA', matricula_resized)
        # cv2.waitKey(0)
        thre_mor = procesa_imagen(matricula_resized)
        ctrs, hierarchy = cv2.findContours(thre_mor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        lista = get_bounding(ctrs, matricula_resized)

        for caracter in lista:
            x, y, w, h = caracter

            recorte = matricula[y:y+h, x:x+w]
            cv2.rectangle(img_color, (x_mat+x, y_mat+y), (x_mat + x + w, y_mat + y + h), (0, 255, 0), 1)

            digito_resized = cv2.resize(recorte, (10, 10), cv2.INTER_LINEAR)
            digitos.append(digito_resized)

    # cv2.imshow('PRUEBA', img_color)
    # cv2.waitKey(0)

    return digitos


def main():
    # Carga de imagenes
    test_imgs = carga_imagenes_carpeta(CARPETA_TEST_FULL_SYSTEM, False)

    # leer carpeta de imagenes -> imagenes, etiquetas
    imagenes_train, etiquetas_train = carga_imagenes_carpeta(CARPETA_TRAINING_OCR, extrae_etiquetas=True)
    caracteristicas_train, clases_train = procesa_ocr_training(imagenes_train, etiquetas_train)
    crf = LinearDiscriminantAnalysis()  # se crea el objeto de entrenador LDA
    cr7_train = crf.fit_transform(caracteristicas_train, clases_train).astype(np.float32)  
    # cr7_test = crf.transform(caracteristicas_test)
    knn_clasif = KNeighborsClassifier(n_neighbors=1)  # se crea el clasificador
    knn_clasif.fit(cr7_train, clases_train)

    for num_img, test_img in enumerate(test_imgs):
        print('PARA LA MATRICULA', num_img)
        digitos_leidos = detecta_digitos(test_img)
        mat_caracteristicas = np.zeros((len(digitos_leidos), 100), dtype=np.uint32)  # matriz de caracteristicas

        for num_digito, digito in enumerate(digitos_leidos):
            print('Digito', num_digito)
            cv2.imshow('ENTRA', digito)
            cv2.waitKey(0)
            vc = digito.flatten()
            mat_caracteristicas[num_digito, :] = vc

        cr_test = crf.transform(mat_caracteristicas)
        digitos_matricula = knn_clasif.predict(cr_test)

        digitos_leidos.clear()

        print(list(map(lambda x: CLASES[x], digitos_matricula)))
        

if __name__ == '__main__':
    main()