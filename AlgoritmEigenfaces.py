import numpy as np
import cv2
from numpy import linalg as la
import statistics
import matplotlib
from matplotlib import pyplot as plt

import math
import random

#mai jos sunt declarate variabile globale
nrPixeli=112*92 #10304
nrTotalPozeAntrenare=8*40 #320
nrPersoane=40
nrPozeAntrenare=8

def AlgoritmulNN(A,pozaCautataVectorizata):
    #stochez distanta din fiecare poza
    nrPozeAntr=320
    z=np.zeros(len(A[0]))
    for i in range(0,len(A[0])):
        z[i]=la.norm(A[:,i]-pozaCautataVectorizata)
        print(z[i])
    pozitiaCautata = np.argmin(z)
    return pozitiaCautata


# matricea de antrenare
A=np.zeros([nrPixeli,nrTotalPozeAntrenare])
caleBD=r'C:\Users\rober\Desktop\probleme\eigenfaces\att_faces'
for i in range(1,nrPersoane+1):
    caleFolderPersoana=caleBD+'\s'+str(i)+'\\'
    for j in range(1,nrPozeAntrenare+1):
        calePozaAntrenare=caleFolderPersoana+str(j)+'.pgm'
        pozaAntrenare=np.array(cv2.imread(calePozaAntrenare,0))
        pozaVectorizata=pozaAntrenare.reshape(-1,)
        A[:,(i-1)*8+j-1]=pozaVectorizata


def inputPozaCautata():
    numarPersoana=input("Numarul persoanei cautate: \n")
    numarPoza=input("Numarul pozei cautate: \n")
    calePozaCautata=r'C:\Users\rober\Desktop\probleme\eigenfaces\att_faces'+'\s'+str(numarPersoana)+'\\'+str(numarPoza)+'.pgm'
    pozaCautata=np.array(cv2.imread(calePozaCautata,0))
    pozaCautataVectorizata=pozaCautata.reshape(-1,)
    return pozaCautataVectorizata



def preprocesare_eigenFaces(matrice,k):
    #calculez poza medie.
    media=np.mean(matrice,axis=1)
    #transpun pt broadcasting
    matrice=(matrice.T-media).T
    #calculez matricea L
    L=np.dot(matrice.T,matrice)
    #calculez vectorii propii ai lui L si ii inmultesc la stanga cu A
    d,v=np.linalg.eig(L)
    #sortez valorile propii cu argsort si ii retin pe cei mai mari k valori
    v=np.dot(matrice,v)
    dSortat=np.argsort(d)
    k=int(k)
    #cei k vectori sunt pastati in High-Quality-Pseudo-Basis pe coloane.
    HQPB=v[:,dSortat[-1:-k-1:-1]]
    #Proiectez pozele din matrice pe HQPB
    proiectii=np.dot(matrice.T,HQPB)
    return [HQPB,proiectii,media]

def eigenFaces(pozaCautataVectorizata,k):
#interogare/cautare

    lista=(preprocesare_eigenFaces(A,k))
    #centrez poza cautata in jurul mediiei
    pozaCautataVectorizata=pozaCautataVectorizata-lista[2]
    #proiectez poza cautata centrata anterior pe HQPB
    pr_cautata=np.dot(pozaCautataVectorizata,lista[0])
#Aplic NN pe proiectia pozei cu celelalte proiectii
    poz=AlgoritmulNN(lista[1].T,pr_cautata)
    return poz


#def procesare_EigenfacesReprezentatiDeClasa(matrice):
   # global dRC,vRC
  #  global dSortatRC
  #  global rc
   # global medie_RC
    #calculez reprez de clasa
   # rc=np.zeros([10304,40])
   # for i in range(0,40):
   #     j=i*8+8
   #     rc[:,i]=np.mean(matrice[:,i*8:j:1],axis=1)
    #calculez poza medie
  #  medie_RC=np.mean(matrice,axis=1)
    #centrez pozele de antrenare
  #  rc=(rc.T-medie_RC).T
  #  #creez matricea de covarianta
  #  matriceCovariantaRC=rc.T@rc
    #calculez v. propii ai matr de cov
   # dRC,vRC=np.linalg.eig(matriceCovariantaRC)
   # vRC=rc@vRC
   # dSortatRC=dRC.argsort()


#def implementare_EigenfacesReprezentantiDeClasa(pozaCautata,catEsteK):
   # k=int(catEsteK)
   # preprocesare=procesare_EigenfacesReprezentatiDeClasa(A)
   ## HQPB=vRC[:,dSortatRC[-1:-k-1:-1]]
   # proiectii =np.dot(rc.T,HQPB)
   ## pozaCautata=pozaCautata-medie_RC
   # pr_cautata=np.dot(pozaCautata,HQPB)
   # poz=AlgoritmulNN(proiectii,pr_cautata)
   # return poz; 



def Lanczos(k2,pozaCautata):
    global medieLanczos
    global hqpbLanczos
    global proiectiiLanczos
    k=int(k2)
    q=np.zeros([10304,k+2])
    q[:,0]=np.zeros(10304)
    q[:,1]=np.ones(10304)
    b=0
    for i in range (1,k):
        w=A*(A.T*q[:,i])-b*q[:,i-1]
        a=(w,q[:,i])
        w=w-a*q[:,i]
        b=la.norm(w)
        q[:,i+1]=w/b
    hqpbLanczos=q[:,2:k+2]
    proiectiiLanczos=np.dot(A.T,hqpbLanczos)
    pozaCautata=pozaCautata-medieLanczos
    proiectie_pozaCautata=np.dot(popozaCautata,hqpbLanczos)
    poz=AlgoritmulNN(proiectiiLanczos.T,proiectie_pozaCautata)


    

    

def plot(pozaCautataInput, poz):
    fig = plt.figure(figsize=(13, 7))
    fig.suptitle('Bontas Robert-Alexandru | Algoritmul Eigenfaces', fontsize=20)
    fig.add_subplot(1, 2, 1)
    plt.imshow(pozaCautataInput.reshape(112,92),cmap='gray',vmin=0,vmax=255)
    plt.axis('off')
    plt.title("Poza introdusa de utilizator")

    fig.add_subplot(1, 2, 2)
    plt.imshow(A[:,poz].reshape(112,92),cmap='gray',vmin=0,vmax=255)
    plt.axis('off')
    plt.title("Poza gasita de algoritm")
    plt.show()

def main():
    i=1
    while i==1:
        print("Scrie eigenfaces pentru algoritmul Eigenfaces sau lanczos pentru Lanczos: ")
        ceAlg=input()
        if ceAlg=="eigenfaces":
                pozaCautataInput=inputPozaCautata()
                catEsteK=input('Introdu K pentru preprocesare: ')
                poz=eigenFaces(pozaCautataInput,catEsteK)
                plot(pozaCautataInput,poz)
        elif ceAlg=="lanczos":
                pozaCautataInput=inputPozaCautata()
                catEsteK=input('Introdu K pentru preprocesare: ')
                poz=Lanczos(catEsteK,pozaCautataInput)
                plot(pozaCautataInput,poz)
        else:
            i=2


main()

