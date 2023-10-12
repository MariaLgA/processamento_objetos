import cv2
import numpy

#leitura da imagem
img = cv2.imread("./imagem_original.jpeg")

#redimensionamento da imagem, proporcao 16:9
new_width = 640
new_heigth = 360
new_size = (new_width,new_heigth)

resize = cv2.resize(img,new_size)

# conversao da imagem de BGR para HSV
img_hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV)

#criacao das mascaras
blue_mask = cv2.inRange(img_hsv,(90,66,153),(130,255,255))
green_mask = cv2.inRange(img_hsv,(57,50,140),(75,255,255))
yellow_mask = cv2.inRange(img_hsv,(22,30,190),(30,255,255))
first_brow_mask= cv2.inRange(img_hsv,(140,15,80),(180,91,215))
second_brow_mask= cv2.inRange(img_hsv,(0,15,75),(9,91,170))
red_mask = cv2.inRange(img_hsv,(150,40,130),(180,255,255))
orange_mask = cv2.inRange(img_hsv,(3,70,100),(13,255,255))

# operacao OR - exemplo
or_mask_examples = cv2.bitwise_or(yellow_mask,green_mask)

# operacao da mascara OR com a mascara marrom
or_mask_brow = cv2.bitwise_or(first_brow_mask,second_brow_mask)

# operacao da mascara AND
and_mask = cv2.bitwise_and(resize,resize,mask = green_mask)

# kernel para usar na erosao e dilatacao
kernel = numpy.ones((3,3), numpy.uint8) 

#opercaos de erosao nas mascaras, pois todas elas tem ruidos (pixels que nao se enquandram com o que queremos)
blue_erode= cv2.erode(blue_mask,kernel,iterations=1)
green_erode = cv2.erode(green_mask,kernel,iterations=1)
yellow_erode = cv2.erode(yellow_mask,kernel,iterations=1)
brow_erode= cv2.erode(or_mask_brow,kernel,iterations=1)
red_erode = cv2.erode(red_mask,kernel,iterations=1)
orange_erode = cv2.erode(orange_mask,kernel,iterations=3)

#operacoes de dilatacao nas mascaras
blue_dilate = cv2.dilate(blue_erode,kernel,iterations=1)
gree_dilate = cv2.dilate(green_erode,kernel,iterations=1)
yellow_dilate = cv2.dilate(yellow_erode,kernel,iterations=1)
brow_dilate = cv2.dilate(brow_erode,kernel,iterations=1)
red_dilate = cv2.dilate(red_erode,kernel,iterations=1)
orange_dilate = cv2.dilate(orange_erode,kernel,iterations=3)

#achando os contornos dos mms conforme as mascaras de cores e escrita na imagem da quantidade de mms de cada cor foram contados
cont =0
name = ["azul","verde","amarelo","marrom","vermelho","laranja"]
color = [(255,0,0),(0,255,0),(0,255,255),(0,75,150),(0,0,255),(0,165,255)]
array = [blue_dilate,gree_dilate,yellow_dilate,brow_dilate,red_dilate,orange_dilate]
for i in range(len(array)):
    qtd = 0
    contornos,hierarquia = cv2.findContours(array[i],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for x in range(len(contornos)):
        area = cv2.contourArea(contornos[x])
        #if somente deve ser usado caso precise filtrar os objetos achados por tamanho
        if(area>400):
            #desenhando os contornos
            desenho_contorno = cv2.drawContours(resize,contornos,x,color[i],2)
            qtd +=1
    cv2.putText(resize,name[i] + ": "+ str(qtd), (10+cont,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1, cv2.LINE_AA)
    cont+=100
        

#output da imagem, waitKey para ficar com a imagem travada na tela ate que algum comando a feche
cv2.imshow("imagem final processada",resize)
cv2.waitKey(0)

#salvar a imagem
cv2.imwrite("./imagem final processada.jpeg",resize)

