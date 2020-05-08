# /usr/bin/python3


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
            x,y,w,h = rect
            caracter_pot = matricula_resized[y:y+h, x:x+w]


def procesa_ocr_training(caracteres_ocr):
    for caracter in caracteres_ocr:
        caracter = cv2.resize(caracter, dsize=(10,10), interpolation=cv2.INTER_LINEAR)
        caracter_gray = cv2.cvtColor(caracter, cv2.COLOR_BGR2GRAY)
        caracter_blur = cv2.GaussianBlur(caracter_gray, (7, 7), 0)
        thresh_inv = cv2.adaptiveThreshold(caracter_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 39, 1)

        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(thresh_inv, cv2.MORPH_DILATE, kernel3)

        ctrs, hierarchy = cv2.findContours(thre_mor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print('PROCESADO')


def main():
    # Carga de imagenes
    test_imgs = carga_imagenes_carpeta(CARPETA_TEST)
    matriculas = detecta_matriculas(test_imgs)
    detecta_digitos(matriculas)

    test_ocr = carga_imagenes_carpeta(CARPETA_TRAIN_OCR)
    procesa_ocr_training(test_ocr)


if __name__ == '__main__':
    main()
