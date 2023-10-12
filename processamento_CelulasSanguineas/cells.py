#import numpy
from matplotlib import pyplot
import cv2
import numpy


#leitura da imagem
img = cv2.imread("/home/ze/Downloads/imagens_cortadas/celulas.jpg")

#convertendo de BGR para HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite("./img_hsv.jpeg",hsv)

#mascara
purple_mask = cv2.inRange(hsv,(130,80,180),(150,255,255))
cv2.imwrite("./mascara_roxa.jpeg",purple_mask)

#operracao AND
and_logic = cv2.bitwise_and(img,img, mask = purple_mask)

cv2.imwrite("./logica_and.jpeg",and_logic)

#operacoes de erosao e dilatacao
kernel = numpy.ones((3,3), numpy.uint8) 

purple_erode = cv2.erode(purple_mask,kernel,iterations=3)

purple_dilate = cv2.dilate(purple_erode,kernel,iterations=5)

cv2.imwrite("./mascara_apos_erosao.jpeg",purple_erode)
cv2.imwrite("./mascara_apos_dilatacao.jpeg",purple_dilate)

#achando os contornos
contador=0
contornos,hierarquia = cv2.findContours(purple_dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contornos)):
    area = cv2.contourArea(contornos[i])
    if(area>1000):
        #desenho dos contornos
        desenho_contorno = cv2.drawContours(img,contornos,i,(0,0,0),2)
        contador+=1
cv2.putText(img,"celulas brancas" + ": "+ str(contador), (40,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1, cv2.LINE_AA)

#output da imagem
cv2.imshow("imagem final processada",img)
cv2.waitKey(0)

cv2.imwrite("./celulas_processadas.jpeg",img)


