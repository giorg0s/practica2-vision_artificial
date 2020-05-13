# /usr/bin/python3
# -*- coding: utf-8 -*-

from carga_imagenes import carga_imagenes_carpeta
from deteccion_haar import procesamiento_img_haar
from preprocesado import *
from entrenamiento_lda import *
from clasificadores import clasificador_knn
from clasificadores import clasificador_bayes

CLASIFICADOR_MATRICULAS = 'assets/haar/matriculas.xml'
CARPETA_TEST_OCR = 'testing_ocr'

cascade_matriculas = cv2.CascadeClassifier(CLASIFICADOR_MATRICULAS)


def detecta_matriculas(imagenes):
    frontales = []  # array con los frontales detectados
    matriculas = []

    for i, img in enumerate(imagenes):
        frontal_coche = procesamiento_img_haar(img)

        cv2.imshow('FRONTAL', frontal_coche)
        cv2.waitKey(0)
        # frontal_coche = cv2.cvtColor(frontal_coche, cv2.COLOR_GRAY2BGR)
        frontales.append(frontal_coche)

    for n, frontal in enumerate(frontales):
        img_procesada = cascade_matriculas.detectMultiScale(frontal, scaleFactor=1.02, minNeighbors=7, minSize=(10, 10))

        if img_procesada is ():
            print('ERROR')
        for (x, y, w, h) in img_procesada:
            imagen_rect = cv2.rectangle(frontal, (x, y), (x + w, y + h), (0, 0, 0), 2)
            matricula = imagen_rect[y:y + h, x:x + w]
            matriculas.append(matricula)

        # cv2.imshow('MATRICULA', matricula)
        # cv2.waitKey(0)

    return matriculas


def detecta_digitos(matriculas):
    digitos = []

    for i, matricula in enumerate(matriculas):
        matricula_resized = cv2.resize(matricula, (matricula.shape[1] * 3, matricula.shape[0] * 3))
        thre_mor = procesa_imagen(matricula)
        ctrs, hierarchy = cv2.findContours(thre_mor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.imshow('RESIZED', matricula)
        cv2.imshow(0)

        lista = get_bounding(ctrs, matricula_resized)

        for rect in lista:
            x, y, w, h = rect
            recorte = matricula_resized[y:y + h, x:x + w]
            digitos.append(recorte)
            
            # cv2.imshow('DIGITO', recorte)
            # cv2.waitKey(0)

    return digitos


def main():
    # Carga de imagenes
    test_imgs = carga_imagenes_carpeta(CARPETA_TEST_OCR, False)
    matriculas = detecta_matriculas(test_imgs)
    digitos = detecta_digitos(matriculas)

    for i, digito in enumerate(digitos):
        print('Estoy usando el digito', i)
        clasificador_knn(digito)

    # clasificador_bayes(cr, np.ndarray.astype(vector_etiquetas, dtype=np.float32))


if __name__ == '__main__':
    main()
