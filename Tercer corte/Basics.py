import face_recognition
import cv2
import numpy as np

# Cargar imágenes (con los nombres REALES de los archivos)
imgzend = face_recognition.load_image_file('ImagesBasic/Zendaya.jpg')
imgTest = face_recognition.load_image_file('ImagesBasic/ZendeyaTest.jpg')

# Convertir colores de RGB (face_recognition) a BGR (OpenCV)
imgzend = cv2.cvtColor(imgzend, cv2.COLOR_RGB2BGR)
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_RGB2BGR)

faceLoc = face_recognition.face_locations(imgzend)[0]
encodezend = face_recognition.face_encodings(imgzend)[0]
cv2.rectangle(imgzend,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodezend],encodeTest)
faceDis = face_recognition.face_distance([encodezend],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

# Mostrar las imágenes
cv2.imshow('Zendaya', imgzend)
cv2.imshow('Zendaya Test', imgTest)
cv2.waitKey(0)


print("¡Éxito! Las imágenes se cargaron correctamente.")