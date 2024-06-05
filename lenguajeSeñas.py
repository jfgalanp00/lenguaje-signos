#-------------------------------------------
# SEGMENTAR LA REGIÓN DE LA MANO DE UNA SECUENCIA DE VIDEO
#-------------------------------------------
import tensorflow as tf 
import h5py

model_path='modelmiguelnumber.h5'

import os
model=tf.keras.models.load_model(model_path)

import cv2
import numpy as np

bg = None

#--------------------------------------------------
# Encontrar el promedio móvil sobre el fondo
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # inicializar el fondo
    if bg is None:
        bg = image.copy().astype("float")
        return

    # calcular el promedio ponderado, acumularlo y actualizar el fondo
    cv2.accumulateWeighted(image, bg, aWeight)

#---------------------------------------------
# Para segmentar la región de la mano en la imagen
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # encontrar la diferencia absoluta entre el fondo y el fotograma actual
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # aplicar umbral a la imagen de diferencia para obtener el primer plano
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # obtener los contornos en la imagen umbralizada
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # devolver Ninguno si no se detectan contornos
    if len(cnts) == 0:
        return
    else:
        # basado en el área del contorno, obtener el contorno máximo que es la mano
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#-----------------
# FUNCIÓN PRINCIPAL
#-----------------
if __name__ == "__main__":
    
    lista=os.listdir('Signos Numeros')
    # inicializar el peso para el promedio móvil
    aWeight = 0.5

    # obtener la referencia a la cámara web
    camera = cv2.VideoCapture(0)

    # coordenadas de la región de interés (ROI)
#    top, right, bottom, left = 50, 400, 225, 590
    
    
# Cambio la posicion del cuadrito
    top, right, bottom, left = 50, 50, 200, 200
    
    # inicializar el número de fotogramas
    num_frames = 0

    # continuar en un bucle hasta que se interrumpa
    while(True):
        # obtener el fotograma actual
        (grabbed, frame) = camera.read()

        # redimensionar el fotograma
        #frame = imutils.resize(frame, width=700)

        # voltear el fotograma para que no sea la vista espejo
        frame = cv2.flip(frame, 1)

        # clonar el fotograma
        clone = frame.copy()

        # obtener la altura y ancho del fotograma
        (height, width) = frame.shape[:2]

        # obtener la ROI
        roi = frame[top:bottom, right:left]
        gray=roi
        # convertir la ROI a escala de grises y desenfocarla
        #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (7, 7), 0)
        k=2
        resized = cv2.resize(roi, (28*k,28*k), interpolation = cv2.INTER_AREA)/255
        
        # para obtener el fondo, seguir buscando hasta que se alcance un umbral
        # para que nuestro modelo de promedio móvil se calibre

        #IMPORTANTE: roi es el cuadradito.

        #Prediccion del modelo
        
        
        pred=model.predict(resized.reshape(-1,28*k,28*k,3))
        #b=np.argmax(b)
        abc = '0123456789'
        
        #x = np.array([4,6,7,3, 1, 8])
        index=np.argsort(pred)
        #print(index)
        
        # Los tres ultimos
        tres=index[-3:][0]
        l3=abc[tres[0]]
        l2=abc[tres[1]]
        l1=abc[tres[2]]
        # Letra correcta:
        #index[-1]
        
        letra=abc[np.argmax(pred)]
        # dibujar la mano segmentada
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(clone, letra, (left-90, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        #Las de abajo
        cv2.putText(clone, l2, (left-150, top+190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(clone, l3, (left-10, top+190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        # incrementar el número de fotogramas
        num_frames += 1

        # mostrar el fotograma con la mano segmentada
        cv2.imshow("Video Feed", clone)


        # si el usuario presiona "q", detener el bucle
        keypress2 = cv2.waitKey(1) 
        if keypress2 == ord(" "):
            letrica=lista[np.random.randint(10)]
            letraimagen=cv2.imread('Signos Numeros/'+ letrica)
            letraimagen=cv2.resize(letraimagen, (150,150), interpolation = cv2.INTER_AREA)
        #cv2.imshow("Letra", letraimagen)
            cloneletrica = letraimagen.copy()
        #cv2.putText(cloneletrica, letrica, (left-100, top+50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            
            # Usando el método cv2.putText()
            letraimagen = cv2.putText(letraimagen, str(letrica[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2, cv2.LINE_AA) 
   
            # Mostrar la imagen
            cv2.imshow("Letra", letraimagen) 
        #    sleep(5)
        # si el usuario presiona "q", detener el bucle       
        keypress = cv2.waitKey(1) 
        if keypress == ord("q"):
            break

# liberar memoria
camera.release()
cv2.destroyAllWindows()
