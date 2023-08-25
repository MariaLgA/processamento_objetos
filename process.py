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
img = cv2.imread("/home/ze/Downloads/imagens_cortadas/mmmistura_org.jpeg")
#img = cv2.imread("/home/ze/Downloads/imagens_cortadas/mm_org.jpeg")


#redimensionar
proporcao = float(img.shape[0]/img.shape[1])
#new_largura = 1024
new_largura = 512
new_altura = int(new_largura*proporcao)
nova_dimensao = (new_largura,new_altura)
redimensionar = cv2.resize(img,nova_dimensao)
print(proporcao,new_altura,nova_dimensao)

cv2.imwrite("/home/ze/Downloads/imagens_cortadas/mmMisturada_contorno.jpeg",redimensionar)


#regra_tres no v e s - 255, no caso/ h vai até 360 degress, então h = h/2
#binarizacao
hsv = cv2.cvtColor(redimensionar, cv2.COLOR_BGR2HSV)


#mascaras
mascara_azul = cv2.inRange(hsv,(90,66,153),(130,255,255))
mascara_verde = cv2.inRange(hsv,(57,50,140),(75,255,255))
mascara_amarelo = cv2.inRange(hsv,(22,30,190),(30,255,255))
mascara_marrom_first= cv2.inRange(hsv,(140,15,80),(180,91,215))
mascara_marrom_second= cv2.inRange(hsv,(0,15,75),(9,91,170))
mascara_vermelho = cv2.inRange(hsv,(150,40,130),(180,255,255))
mascara_laranja = cv2.inRange(hsv,(3,70,100),(13,255,255))

#mascara_marrom_first= cv2.inRange(hsv,(140,15,80),(180,75,215))
#mascara_marrom_second= cv2.inRange(hsv,(0,15,90),(9,75,170))

cv2.imwrite("/home/ze/Downloads/imagens_cortadas/mm_mascaraMarromFirst.jpeg",mascara_marrom_first)

#mascara OR
m_marrom = cv2.bitwise_or(mascara_marrom_first,mascara_marrom_second)

#mascara AND
#m = cv2.bitwise_and(redimension￼ar,redimensionar, mask = m_marrom)

#cv2.imwrite("/home/ze/Downloads/imagens_cortadas/mm_mascaraMarromFirst.jpeg",m)


#filtros erosao
kernel = numpy.ones((3,3), numpy.uint8) 
erosao_azul= cv2.erode(mascara_azul,kernel,iterations=1)
erosao_verde = cv2.erode(mascara_verde,kernel,iterations=1)
erosao_amarelo = cv2.erode(mascara_amarelo,kernel,iterations=1)
erosao_marrom= cv2.erode(m_marrom,kernel,iterations=1)
erosao_vermelho = cv2.erode(mascara_vermelho,kernel,iterations=1)
erosao_laranja = cv2.erode(mascara_laranja,kernel,iterations=3)

#cv2.imwrite("/home/ze/Downloads/imagens_cortadas/mm_erosaoAzul.jpeg",erosao_marrom)

#filtros erosao e dilatacao
dilatacao_azul = cv2.dilate(erosao_azul,kernel,iterations=1)
dilatacao_verde = cv2.dilate(erosao_verde,kernel,iterations=1)
dilatacao_amarelo = cv2.dilate(erosao_amarelo,kernel,iterations=1)
dilatacao_marrom = cv2.dilate(erosao_marrom,kernel,iterations=1)
dilatacao_vermelho = cv2.dilate(erosao_vermelho,kernel,iterations=1)
dilatacao_laranja = cv2.dilate(erosao_laranja,kernel,iterations=3)

cv2.imwrite("/home/ze/Downloads/imagens_cortadas/mm_dilatacaoAzul.jpeg",dilatacao_marrom)


#achando os contornos
cont =0
nome = ["azul","verde","amarelo","marrom","vermelho","laranja"]
vetor = [dilatacao_azul,dilatacao_verde,dilatacao_amarelo,dilatacao_marrom,dilatacao_vermelho,dilatacao_laranja]
for i in range(len(vetor)):
    quantidade = 0
    contornos,hierarquia = cv2.findContours(vetor[i],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for x in range(len(contornos)):
        area = cv2.contourArea(contornos[x])
        if(area>400):
            #desenhando os contornos
            desenho_contorno = cv2.drawContours(redimensionar,contornos,x,(0,0,0),2)
            quantidade +=1
    cv2.putText(redimensionar,nome[i] + ": "+ str(quantidade), (10+cont,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1, cv2.LINE_AA)
    cont+=100
        



# #output da imagem


cv2.imshow("teste",redimensionar)
cv2.imwrite("/home/ze/Downloads/imagens_cortadas/contornoMisturado.jpeg",redimensionar)
#cv2.imshow("300",m_marrom)
#cv2.imshow("menor",mascara_marrom_second)
# # cv2.imshow("amarelo",mascara_amarelo)
# #cv2.imshow("hsv",mascara_azul)

#tempo de espera
cv2.waitKey(0)