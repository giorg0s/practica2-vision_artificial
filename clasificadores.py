# /usr/bin/python3

def clasificador_knn(mat_cr, vec_e):
    knn_clasif = cv2.ml.KNearest_create() # se crea el clasificador
    # knn_clasif.train(mat_cr, vec_e)

    # ret, result, neighbours, dist = knn_clasif.findNearest(test, k=5)

    print('hola')
