# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f_x_ex1(x):
    return((x**3)-np.cos(x))

#Agora irei definir uma função que determina se um número é negativo ou não
#afim de comparar os sinais entre dois números

def sinal_negativo(x):
    if(x<0):
        sinal=True
    else:
        sinal=False
    return(sinal)

def erro_relativo(a,b):
    return((a-b)/a)

def resol_matr_trian(matriz_t_s):
    linhas = np.shape(matriz_t_s)[0]
    vet_sol = np.zeros(linhas)
    for i in range(1,linhas+1):          
        aux=matriz_t_s[linhas-i][linhas] #Note que aqui, estamos percorrendo o #vetor para tras
        for j in range(1,i+1): #A cada iteracao, incrementamos o valor de aux  #com as solucoes ja obtidas
            aux += -matriz_t_s[linhas-i][linhas-j]*vet_sol[linhas-j] 
        vet_sol[linhas-i] = aux/matriz_t_s[linhas-i][linhas-i] 
    return(vet_sol)
"""
Created on Wed Oct  7 19:31:51 2020

@author: lucas
"""

"""
Repetir o item A com método Newton-Raphson
"""

#Obs: o indice _ex2 é referencia ao exercício 2.

#Primeiro irei definir f_x_ex2, que recebe um numero real x e retorna x³ - cos(x)
def f_x_ex2(x):
    return(np.sin(x)-(np.e)**(-x))

#Agora irei definir a função que equivale à primeira derivada de f_x_ex2
def df_x_ex_2(x):
    return(np.cos(x)+(np.e)**(-x))

#Irei definir também a função g_x, pertencente ao método de Newton-Raphson
def g_x_ex_2(x):
    return( x - ( f_x_ex2(x) / df_x_ex_2(x)) ) 

E_ex2 = 1.e-6

x_n_ex_2 = 0 #Número suficientemente próximo da raíz

n = 1 

columns = ['n',
'Xn',
'f(Xn)',
'f'+chr(39)+'(Xn)',
'Erro Relativo']

tabela_ex_2 = pd.DataFrame(columns = columns)

erro = erro_relativo(g_x_ex_2(x_n_ex_2),x_n_ex_2)

while ( erro_relativo(g_x_ex_2(x_n_ex_2),x_n_ex_2) > E_ex2 ):
    erro = erro_relativo(g_x_ex_2(x_n_ex_2),x_n_ex_2)
    #A cada iteração, df2 adiciona uma linha, dada por df2
    df2 = pd.DataFrame([[n,x_n_ex_2,f_x_ex2(x_n_ex_2),df_x_ex_2(x_n_ex_2),
                         erro]],columns = columns)
    
    print(n,x_n_ex_2,f_x_ex2(x_n_ex_2),df_x_ex_2(x_n_ex_2),erro)
    tabela_ex_2 = tabela_ex_2.append(df2, ignore_index=True)
    x_n_ex_2 = g_x_ex_2(x_n_ex_2)
    n += 1
    
tabela_ex_2.to_csv('tabela_B_EP1.csv',index=False)