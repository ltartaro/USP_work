# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 20:20:40 2020

@author: lucas
"""

#%% ----------------------- I- EULER E RUNGE-KUTTA ----------------------------
import numpy as np
import matplotlib.pyplot as plt
"""
Resolver a equação diferencial ordinária de 2a. ordem:
    y''=12t^2+4t^3-t^4+y-y'
    
    Com as seguintes condições iniciais:
        y(0) = 0
        y'(0) = 0
    
    A equação pode ser reescrita como:
        z' = g(t,y,z) = 12t^2+4t^3-t^4+y-z
        y' = z
    
    Calcular y(5) e dy/dt(t=5) usando h = 0.001
    
    Comparar com o resultado exato, y = t^4
"""

"""
MÉTODO DE EULER PARA EDO DE 2a. ORDEM
-Primeiro irei definir g(t,y,z)
"""

def g_tyz(t,y,z):
    return((12*(t**2)) + (4*(t**3)) - (t**4)  +  y  -  z  )

"""
Agora será gerada uma rotina que atualiza z_(i+1) e y_(i+1) em função dos 
valores z_i e y_i
"""
#Note que t já é incrementado em h
    
def euler_2a_ordem(h=0.001,t_0=0,y_0=0,z_0=0,t_fin=5,g_tyz=g_tyz):
    t = t_0
    y_i = y_0
    z_i = z_0
    #Para o proximo looping, irei ficar atualizando os valores de y_0 e z_0
    while(t<=t_fin):
        z_i = z_0 + h*(g_tyz(t,y_0,z_0))
        y_i = y_0 + h*z_i
        t = t + h
        z_0 = z_i;y_0=y_i;
        print(z_i)
        print(y_i)
        print(t)
"""
Criando rotina para o RK4
-Primeiro irei definir a subrotina dada no EP4
-Irei adicionar a definição de g_tyz
"""
def subrot_rk4(t_i,y_i,z_i,h=0.001,g_tyz=g_tyz):
    k_1y = h*z_i
    k_1z = h*g_tyz(t_i,       y_i,            z_i)
    k_2y = h*(z_i+(k_1z/2))
    k_2z = h*g_tyz(t_i+(h/2), y_i + (k_1y/2), z_i +(k_1z/2))
    k_3y = h*(z_i+(k_2z/2))
    k_3z = h*g_tyz(t_i+(h/2), y_i + (k_2y/2), z_i +(k_2z/2))
    k_4y = h*(z_i + k_3z)
    k_4z = h*g_tyz(t_i+h,     y_i+   k_3y   , z_i + k_3z )
    
    y_i = y_i + (k_1y + (2*k_2y) + (2*k_3y) + k_4y)/6
    z_i = z_i + (k_1z + (2*k_2z) + (2*k_3z) + k_4z)/6
    
    t_i = t_i + h
    return (t_i,y_i,z_i)

def rk4(h=0.001,t_0=0,y_0=0,z_0=0,t_fin=5,g_tyz=g_tyz):
    t_i = t_0
    y_i = y_0
    z_i = z_0
    matrix_tyz = np.zeros(([3,int(t_fin/h)+1]))
    i = 0
    while(t_i<=t_fin):
        t_i,y_i,z_i = subrot_rk4(t_i,y_i,z_i,h,g_tyz)
        print(z_i)
        print(y_i)
        print(t_i)
        matrix_tyz[0][i] = t_i
        matrix_tyz[1][i] = y_i
        matrix_tyz[2][i] = z_i
        i+=1
    return(matrix_tyz)
        
#%% ---------------------- II - EQUACAO DE DUFFING ----------------------------
# EQUAÇÃO DE DUFFING, "DOUBLE WELL POTENTIAL"
#ITEM A
"""
#Definindo a funcao para o potencial de duplo poço:
    x''-0.5*x(1-x^2) = 0 implica que x'' = 0.5*x(1-x^2)
    logo g_txz = 0.5*x(1-x^2)
"""

def g_poco_dup(t,y,z):
    return( 0.5*y*(1-(y**2)))
           
a = rk4(h=0.001,t_0=0,y_0=1,z_0=-1,t_fin=9,g_tyz=g_poco_dup)

fig = plt.figure()
fig = plt.plot(a[1,:-1],a[2,:-1])
a = rk4(h=0.001,t_0=0,y_0=1,z_0=-0.1,t_fin=9,g_tyz=g_poco_dup)
fig = plt.plot(a[1,:-1],a[2,:-1])

a = rk4(h=0.001,t_0=0,y_0=1,z_0=-0.50001,t_fin=56,g_tyz=g_poco_dup)
fig = plt.plot(a[1,:-1],a[2,:-1])

#%% ITEM B

"""
#Definindo a funcao para o potencial de duplo poço com amortecimento:
    x''+2*gama*x'-0.5*x(1-x^2) = 0 implica que x'' = 0.5*x(1-x^2)
    logo g_txz = 0.5*x(1-x^2) - 2*gama*x'
    Pela definição, x' = z:
         g_txz = 0.5*x(1-x^2) - 2*gama*z
"""

def g_poco_dup_amort_g025(t,y,z):
    return( 0.5*y*(1-(y**2))-0.25*z)

a = rk4(h=0.001,t_0=0,y_0=1,z_0=-1,t_fin=50,g_tyz=g_poco_dup_amort_g025)

fig_1 = plt.figure()
fig_1 = plt.plot(a[1,:-1],a[2,:-1])


def g_poco_dup_amort_gama08(t,y,z):
    return( 0.5*y*(1-(y**2))-0.8*z)

a = rk4(h=0.001,t_0=0,y_0=1,z_0=-1,t_fin=50,g_tyz=g_poco_dup_amort_gama08)

fig_1 = plt.plot(a[1,:-1],a[2,:-1])

#%% ITEM C

"""
Defininindo a funcao do item b forcada:
     x''+0.25*x'-0.5*x(1-x^2) = F cos(wt) implica que, com w = 1
         x''= F*cos(t) - 0.25*x' + 0.5*x(1-x^2)
    Pela definição, x' = z:
         g_txz = 0.5*x(1-x^2) - 0.25*z + F*cos(t)
"""

def g_poco_amort_forc(t,y,z,F=0.22):
    return(0.5*y*(1-(y**2))-0.25*z + F*np.cos(t))

a = rk4(h=0.001,t_0=0,y_0=1,z_0=-1,t_fin=100,g_tyz=g_poco_amort_forc)

fig_1 = plt.figure()
fig_1 = plt.plot(a[1,5000:-1],a[2,5000:-1],label='F=0.22')


def g_poco_amort_forc(t,y,z,F=0.23):
    return(0.5*y*(1-(y**2))-0.25*z + F*np.cos(t))

a = rk4(h=0.001,t_0=0,y_0=1,z_0=-1,t_fin=100,g_tyz=g_poco_amort_forc)

fig_1 = plt.plot(a[1,5000:-1],a[2,5000:-1],label='F=0.23')

def g_poco_amort_forc(t,y,z,F=0.28):
    return(0.5*y*(1-(y**2))-0.25*z + F*np.cos(t))

a = rk4(h=0.001,t_0=0,y_0=1,z_0=-1,t_fin=100,g_tyz=g_poco_amort_forc)

fig_1 = plt.plot(a[1,5000:-1],a[2,5000:-1],label='F=0.28')

def g_poco_amort_forc(t,y,z,F=0.35):
    return(0.5*y*(1-(y**2))-0.25*z + F*np.cos(t))

a = rk4(h=0.001,t_0=0,y_0=1,z_0=-1,t_fin=100,g_tyz=g_poco_amort_forc)

fig_1 = plt.plot(a[1,5000:-1],a[2,5000:-1],label='F=0.35')

def g_poco_amort_forc(t,y,z,F=0.6):
    return(0.5*y*(1-(y**2))-0.25*z + F*np.cos(t))

a = rk4(h=0.001,t_0=0,y_0=1,z_0=-1,t_fin=100,g_tyz=g_poco_amort_forc)

fig_1 = plt.plot(a[1,50000:-1],a[2,50000:-1],label='F=0.60')

fig_1 = plt.legend()

#%% ---------------------- 2) DIAGRAMA DE BIFURCAÇÃO --------------------------
