# /usr/bin/python3

import cv2
import time
import numpy as np

CARPETA_TEST = 'img/test'
CLASIFICADOR = 'assets/haar/coches.xml'

# Se crea el cascade classifier
cascade = cv2.CascadeClassifier(CLASIFICADOR)


def procesamiento_img_haar(imagen):
    # Se convierte la imagen a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # El código se basa en los ejemplos que se encuentran en la web: https://www.programcreek.com/python/example/79435/cv2.CascadeClassifier
    imagen_eq = cv2.equalizeHist(gray)

    imagen_gris = cascade.detectMultiScale(imagen_eq, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))

    if imagen_gris is ():
        print('ERROR')
    for (x, y, w, h) in imagen_gris:
        imagen_gris = cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 0, 0), 2)
        frontal_detectado = imagen_gris[y:y+h, x:x+w]

        return frontal_detectado

    # cv2.imwrite(os.path.join('img', 'output', 'output_haar'+str(contador)+'.png'), imagen)

    # cv2.destroyAllWindows()


def detector_coches(imagenes):
    tiempos = []
    for i, img in enumerate(imagenes):
        inicio = time.time()
        print("PROCESANDO IMAGEN", i)
        procesamiento_img_haar(img, i)
        fin = time.time()
        tiempos.append(fin-inicio)

    print('TIEMPO MEDIO POR IMAGEN', sum(tiempos)/len(imagenes))

