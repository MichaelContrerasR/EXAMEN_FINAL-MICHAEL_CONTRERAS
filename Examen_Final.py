#-----------------------------------------
# EXAMEN FINAL: MICHAEL CONTRERAS RAMIREZ
#-----------------------------------------
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
""" EXAMEN FINAL 
"""
##### Cargar imagen
if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
##### punto # 1 conteo pixeles del campo y porcentaje
    # Hue histogram
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])

    # Hue histogram max and location of max
    max_val = hist_hue.max()
    max_pos = int(hist_hue.argmax())

    # Peak mask
    lim_inf = (max_pos - 10, 0, 0)
    lim_sup = (max_pos + 10, 255, 255)
    mask = cv2.inRange(image_hsv, lim_inf, lim_sup)
    mask_not = cv2.bitwise_not(mask)

    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    PixT = height * width
    ## conteo de pixeles del campo
    numpixv= np.sum(mask == 255)
    # Porcentaje de pixeles
    PixVer = (numpixv / PixT)*100
    print('Numero total de pixels:', PixT)
    print('Number de pixeles pertenecientes al campo:', numpixv)
    print('El porcentaje de pixeles verdes en el campo es de:',  PixVer)

    cv2.namedWindow("Mascara de Imagen pixeles campo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mascara de Imagen pixeles campo", 1280, 720)
    cv2.imshow("Mascara de Imagen pixeles campo", mask)
    cv2.waitKey(0)
    canny = cv2.Canny(mask, 50, 150)

    ### Punto # 2 Encontrar Jugadores sobre el campo
    # Decteccion de contornos
    contours, hierarchy = cv2.findContours(mask_not, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contJ = 0
    # Ubicacion de punto central, caculo de area y perimetro de cada objeto encontrado
    for i in range(len(contours)):
        if len(contours[i]) > 3:
            cont = contours[i]
            area = cv2.contourArea(cont)
            if area > 1000 and area < 5000:
                M = cv2.moments(cont)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                x, y, width, height = cv2.boundingRect(contours[i])
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
                contJ = contJ + 1

    Poracier = (contJ / 21) * 100
    print("Se han encontrado {} jugadores en el campo".format(contJ))
    print('El porcentaje correctos de jugadores encontrado en el campo es de:', Poracier)

    cv2.namedWindow("Mascara de Jugadores en el campo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mascara de Jugadores en el campo", 1280, 720)
    cv2.imshow("Mascara de Jugadores en el campo", mask_not)
    cv2.waitKey(0)
    cv2.namedWindow("Imagen Jugadores Ubicados en el campo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Imagen Jugadores Ubicados en el campo", 1280, 720)
    cv2.imshow("Imagen Jugadores Ubicados en el campo", image)
    cv2.waitKey(0)

    # fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    # axs[0, 0].imshow(image)
    # axs[0, 0].set_title('Imagen Original', fontweight="bold")
    # axs[1, 0].imshow(mask)
    # axs[1, 0].set_title('Mascara imagen', fontweight="bold")
    # axs[0, 1].imshow(mask_not)
    # axs[0, 1].set_title('Imagen escala grises', fontweight="bold")
    # axs[1, 1].imshow(canny)
    # axs[1, 1].set_title('Imagen deteccion de bordes', fontweight="bold")
    # plt.show()