#import numpy
from matplotlib import pyplot
import cv2
import numpy


#Threshold
#Onde img>=T fica branco, senão pretocv2.
# img_cinza = cv2.cvtColor(redimensionar,cv2.COLOR_BGR2GRAY)
# T_valor_limite,img_binarizacao = cv2.threshold(img_cinza,50,255,cv2.THRESH_BINARY)


# histograma = cv2.calcHist(img,[0],None,[256],[0,256])
# pyplot.figure()
# pyplot.plot(histograma)
# pyplot.xlim([0,256])
# pyplot.show()


#lendo a imagem
img = cv2.imread("/home/ze/Downloads/imagens_cortadas/celulas.jpg")

#redimensionar
# proporcao = float(img.shape[0]/img.shape[1])
# new_largura = 1024
# new_altura = int(new_largura*proporcao)
# nova_dimensao = (new_largura,new_altura)
# redimensionar = cv2.resize(img,nova_dimensao)
# print(proporcao,new_altura,nova_dimensao)

#regra_tres no v e s - 255, no caso/ h vai até 360 degress, então h = h/2
#binarizacao
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imwrite("/home/ze/Downloads/imagens_cortadas/celula_hsv.jpeg",hsv)


#mascaras
mascara_roxa = cv2.inRange(hsv,(130,80,180),(150,255,255))
cv2.imwrite("/home/ze/Downloads/imagens_cortadas/celula_m.jpeg",mascara_roxa)

#mascara_roxa = cv2.inRange(hsv,(130,90,160),(150,255,220))

#mascara AND
m = cv2.bitwise_and(img,img, mask = mascara_roxa)

cv2.imwrite("/home/ze/Downloads/imagens_cortadas/celula_mascara.jpeg",m)

#filtros erosao e dilatacao
kernel = numpy.ones((3,3), numpy.uint8) 
erosao_roxa = cv2.erode(mascara_roxa,kernel,iterations=3)

dilatacao_roxa = cv2.dilate(erosao_roxa,kernel,iterations=5)

cv2.imwrite("/home/ze/Downloads/imagens_cortadas/celula_erosao.jpeg",dilatacao_roxa)

#achando os contornos
contador=0
contornos,hierarquia = cv2.findContours(dilatacao_roxa,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contornos)):
    area = cv2.contourArea(contornos[i])
    if(area>1000):
        #desenho dos contornos
        desenho_contorno = cv2.drawContours(img,contornos,i,(0,0,0),2)
        contador+=1

cv2.putText(img,"celulas brancas" + ": "+ str(contador), (40,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1, cv2.LINE_AA)
#output da imagem
cv2.imwrite("/home/ze/Downloads/imagens_cortadas/celula_dilatacao.jpeg",img)


cv2.imshow("teste",img)
# cv2.imshow("laranja",mascara_laranja)
# cv2.imshow("verde",mascara_verde)
# cv2.imshow("amarelo",mascara_amarelo)
#cv2.imshow("hsv",mascara_azul)

#tempo de espera
cv2.waitKey(0)