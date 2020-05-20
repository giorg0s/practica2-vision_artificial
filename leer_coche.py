# /usr/bin/python3
# -*- coding: utf-8 -*-

from deteccion_haar import procesamiento_img_haar
from preprocesado import *
from clasificadores import *

CLASIFICADOR_FRONTALES = 'assets/haar/coches.xml'
CLASIFICADOR_MATRICULAS = 'assets/haar/matriculas.xml'
CARPETA_TEST_OCR = 'testing_ocr'
CARPETA_TEST_FULL_SYSTEM = 'testing_full_system'

cascade_frontales = cv2.CascadeClassifier(CLASIFICADOR_FRONTALES)
cascade_matriculas = cv2.CascadeClassifier(CLASIFICADOR_MATRICULAS)
digitos = []


def detecta_digitos(test_img):
    lista_matriculas = []
    lista = []
    img_trabajo = test_img.copy()

    img_color = cv2.cvtColor(img_trabajo, cv2.COLOR_GRAY2BGR)

    frontales = cascade_frontales.detectMultiScale(img_trabajo, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in frontales:
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
        frontal_coche = img_trabajo[y:y + h, x:x + w]

    matriculas = cascade_matriculas.detectMultiScale(img_trabajo, scaleFactor=1.2, minNeighbors=4, minSize=(10, 10),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in matriculas:
        cv2.rectangle(img_color, (x, y), (x+w, y+h), (255, 0, 255), 1)
        matricula = img_trabajo[y:y + h, x:x + w]  # Trozo que corresponde con la matrícula recortada
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

            recorte = matricula_resized[y:y+h, x:x+w]

            cv2.rectangle(img_color, (x, y), (x+w, y + h), (0, 255, 0), 2)

            digito_resized = cv2.resize(recorte, (10, 10), cv2.INTER_LINEAR)
            digitos.append(digito_resized)

    cv2.imshow('PRUEBA', img_color)
    cv2.waitKey(0)

    return lista


def main():
    # Carga de imagenes
    test_imgs = carga_imagenes_carpeta(CARPETA_TEST_OCR, False)

    for test_img in test_imgs:
        digitos_leidos = detecta_digitos(test_img)
        print(len(digitos_leidos))

    clasificador_knn = preparar_clasificador_knn()


if __name__ == '__main__':
    main()
