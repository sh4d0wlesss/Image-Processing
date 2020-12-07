# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:04:47 2019

@author: Faruk Arslan
"""
import cv2 as cv
import numpy as np



def save_file():
    get = 0
    count =0
    while get == 0:
        image_name='images_data/'+str(count+1)+'.jpg'
        image = cv.imread(image_name)
        if image is None:
            print('Tüm resimler okundu')
            get = 1
        else:
            print(str(count+1)+'.jpg resmi okunuyor, histogram olusturuluyor.')
            filename='histograms/'+str(count)+'.npy'
            np.save(filename,get_histogram(image))
            count = count + 1
    return count
########################################  
def get_histogram(img):
    hist = np.zeros((4, 256), dtype=np.double)
    hist[0],hist[1],hist[2]=rgb_hist(img)
    print('------------------')
    hist[3]=lbp(img)
    return hist
########################################

#########################################
def make_distance_list(number_of_hist,image_name,flag):#flag=0 ise rgb, flag=1 ise lbp
    image=cv.imread(image_name,1)
    if image is None:
        print('Resim bulunamadı!')
        return -1
    
    image_hist=get_histogram(image)
    distancelist=np.zeros(number_of_hist, dtype=np.double)
    for i in range(0,number_of_hist):
        filename='histograms/'+str(i)+'.npy'
        saved_hist=np.load(filename)
        distancelist[i]=distance(saved_hist,image_hist,flag)
        
    similar_images_list=find_min_index(distancelist,number_of_hist)
    return similar_images_list
##########################################
def distance(hist1,hist2,flag): #flag=0 ise rgb, flag=1 ise lbp
    if flag==0:
        dist=0
        for i in range(0,256):
            dist = dist + abs(hist1[0][i] - hist2[0][i]) + abs(hist1[1][i] - hist2[1][i]) + abs(hist1[2][i] - hist2[2][i])
    else:
        dist=0
        for i in range(0,256):
            dist = dist + abs(hist1[3][i] - hist2[3][i])
        
    return dist
#########################################
def find_min_index(distancelist,size):
    index_list=[]
    for i in range(0,5):
        min_index=0
        for j in range(i+1,size):
            if distancelist[j] < distancelist[min_index]:
                min_index=j
        index_list.append(min_index)
        distancelist[min_index]=200
        
    return index_list
#########################################
def main():
    exist_file = input('Eğer histogram dosyalari mevcut ise 0 a, mevcut değil ise 1 e basınız: ')
    if int(exist_file) == 1:
        number_of_hist = save_file()
    elif int(exist_file) == 0:
        number_of_hist = int(input('Mevcut histogram dosyası sayısını giriniz: '))
    
    image_name='images_test/'+input('Test etmek istedigin resmin ismini ver: ')
    while image_name is not None:
        similar_image_index_list = make_distance_list(number_of_hist,image_name,0) # 0 verdik ki rgb yapsın
        
        if similar_image_index_list != -1:
            print('RGB benzerliği olan resimler')
            for i in similar_image_index_list:
                print('images_data/'+str(i+1)+'.jpg')
                
            similar_image_index_list = make_distance_list(number_of_hist,image_name,1) # 1 verdik ki rgb yapsın
            print('LBP benzerliği olan resimler')
            for i in similar_image_index_list:
                print('images_data/'+str(i+1)+'.jpg')
        
        image_name='images_test/'+input('Test etmek istedigin resmin ismini ver: ')
    
###########################################
############################# r,g,b değerlerine göre histogram cıkaran fonksiyon
def rgb_hist(img): 
    rows = img.shape[0]
    cols = img.shape[1]
    r_array=np.zeros(256,dtype=np.double)
    g_array=np.zeros(256,dtype=np.double)
    b_array=np.zeros(256,dtype=np.double)

    for i in range(rows): # histogram oluşturulur
        for j in range(cols):
            r_array[img[i][j][2]] += 1
            g_array[img[i][j][1]] += 1
            b_array[img[i][j][0]] += 1
    
    for i in range(256): # normalizasyon yapılır
        r_array[i]=r_array[i]/(rows*cols)
        g_array[i]=g_array[i]/(rows*cols)
        b_array[i]=b_array[i]/(rows*cols)
        
    return r_array,g_array,b_array; # tuple olarak dizileri döndürürüz
########################### local binary partition değerlerine göre histogram cıkarır (ayrıca gray level dönüşüm fonksiyonu yazılmadı,bu fonksiyonun içinde hesaplandı)
def lbp(img):
    rows = img.shape[0]
    cols = img.shape[1]
    lbp_array =np.zeros(256,dtype=np.double)
    x=0
    
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            current_pixel_value = (img[i][j][2]*0.299 + img[i][j][1]*0.587 + img[i][j][0]*0.114 )
            x=0
            if ((img[i-1][j-1][2]*0.299 + img[i-1][j-1][1]*0.587 + img[i-1][j-1][0]*0.114 ) > current_pixel_value):
                x= x+1 # sol üste baktı
            x=x*2
            if ((img[i-1][j][2]*0.299 + img[i-1][j][1]*0.587 + img[i-1][j][0]*0.114 ) > current_pixel_value):
                x= x+1 #üste baktı
            x=x*2
            if ((img[i-1][j+1][2]*0.299 + img[i-1][j+1][1]*0.587 + img[i-1][j+1][0]*0.114 ) > current_pixel_value):
                x= x+1 #sag üste baktı
            x=x*2
            if ((img[i][j-1][2]*0.299 + img[i][j-1][1]*0.587 + img[i][j-1][0]*0.114 ) > current_pixel_value):
                x= x+1 #sola baktı
            x=x*2
            if ((img[i][j+1][2]*0.299 + img[i][j+1][1]*0.587 + img[i][j+1][0]*0.114 ) > current_pixel_value):
                x= x+1 #saga baktı
            x=x*2
            if ((img[i+1][j-1][2]*0.299 + img[i+1][j-1][1]*0.587 + img[i+1][j-1][0]*0.114 ) > current_pixel_value):
                x= x+1 #sol alta baktı
            x=x*2
            if ((img[i+1][j][2]*0.299 + img[i+1][j][1]*0.587 + img[i+1][j][0]*0.114 ) > current_pixel_value):
                x= x+1 #alta baktı
            x=x*2
            if ((img[i+1][j+1][2]*0.299 + img[i+1][j+1][1]*0.587 + img[i+1][j+1][0]*0.114 ) > current_pixel_value):
                x= x+1 #sag alta baktı    
            
            lbp_array[x] += 1
            
    for i in range(256):
        lbp_array[i]=lbp_array[i]/(rows*cols)
            
    #print(lbp_array)
    return lbp_array;  
#######################################################################

main()
