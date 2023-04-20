# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:03:10 2020

@author: lucas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%    EXERCICIO 2

"""
Some 10000 vezes no computador o valor 0.0001 e imprima o resultado com sete decimais
(precisão simples).
"""

#%% precisão simples: np.float32()

valor_p_simp = np.float32(0.0001)
res = np.float32(0.0001)
for i in range(9999):
    res += valor_p_simp
    print(res)
    print(i)
    
#%% precisao dupla: np.float64()
valor_p_simp = np.float64(0.0001)
res = np.float64(0.0001)
for i in range(9999):
    res += valor_p_simp
    print(res)
    print(i)

#%%   EXERCICIO 3
"""
Calcule no computador o maior n! possível até causar over
ow (precisão simples e dupla). Imprima n e n!
"""

i_p_s = np.float32(1)
aux_p_s = np.float32(1)

i_p_d = np.float64(1)
aux_p_d = np.float64(1)
while(aux_p_d != np.inf):
    aux_p_s *= i_p_s
    aux_p_d *= i_p_d
    i_p_s += np.float32(1)
    i_p_d += np.float64(1)
    print("precisao simples","   ",i_p_s,"   ",aux_p_s)
    print("precisao dupla  ","   ",i_p_d,"   ",aux_p_d)
    
#%%  EXERCICIO 4

"""
Calcule com uma calculadora não programável a raiz positiva de x² - 5 = 0
usando o método de bissecção, até o intervalo ser ≤ 0.001.
"""

"""
Para isso vou usar uma simples análise de sinal da função f(x)= x²-5 para x_1,
x_2 e x_m, de forma que:
    se sinal x_1 = sinal x_m então x_1=x_m, se não, x_2 = x_m
    
É esperado alcançar raiz(5) como resposta. Sei que a inversão de sinal ocorre
entre 2 e 3.
"""

def f_x(x):
    return((x**2)-5)

def signal(x):
    if(x<0):
        x=-1
    else:
        x=1
    return(x)

#Escolha de numeros que configuram um intervalo que contem a raiz
#Fazendo o gráfico. Melhor maneira é o gráfico.
    
#Como o intervalo ta sempre diminuindo pela metade, o erro tbm esta sempre dividindo por dois
#ENTAO VEJA: e_n+1 = 0.5e_n
x_1 = 2
x_2 = 3
E=0.00001

while(np.absolute(x_1 - x_2) >= E): #Aqui podia fazer erro relativo (x_1-x_2)/x_2
    x_m = (x_1+x_2)/2
    
    if(signal(f_x(x_1))==signal(f_x(x_m))):
        x_1=x_m
    else:
        x_2=x_m
    
    print(x_m)


#A vantagem desse metodo de bisseccao é que é um metodo bastante seguro
#ele, sempre vai chegar numa raiz.
#A desvantagem é que o metodo é super lento.

#%%   EXERCICIO 5

"""
As vezes a resposta do problema depende do valor inicial.
Funcao iterativa: X_n+1 = G(X_n)

A eq. x² − 3x + 2 = 0 pode ser escrita como x = G(x) em diversas formas para aplicação
do método de substituições sucessivas. Determine analiticamente a região de convergência
para as raízes x = 1, 2 e faça os grácos de convergência y = G(x) superposto à reta y = x
para os seguintes casos:

a) X_n+1 = (X_n² + 2)/3
    
b) X_n+1 = sqrt(3X_n - 2)

Observemos que bastou isolar o x de maneiras diferentes.    

IMPORTANTE: |G'(x)| < 1

"""

def G_x_a(x):
    return((x**2+2)/3)

#-3/2 < x < 3/2

def G_x_b(x):
    return(np.sqrt(3*x-2))

# x > 17/12



fig_5_a = plt.figure(figsize=(8,8),dpi=100)

axis_x = np.arange(0.5,1.5,0.0001)

fig_5_a = plt.plot(axis_x,axis_x)

fig_5_a = plt.plot(axis_x,G_x_a(axis_x))

x_5_a = -3/4
while (np.absolute(x_5_a-G_x_a(x_5_a))>0.000001):
    aux = x_5_a
    x_5_a = G_x_a(x_5_a)
    fig_5_a = plt.plot((x_5_a,aux),(x_5_a,x_5_a),c='k')
    fig_5_a = plt.plot((x_5_a,x_5_a),(G_x_a(x_5_a),x_5_a),c='k')
    print(x_5_a)

fig_5_b = plt.figure(figsize=(8,8),dpi=100)

axis_x = np.arange(0.5,4,0.0001)

fig_5_b = plt.plot(axis_x,axis_x)

fig_5_b = plt.plot(axis_x,G_x_b(axis_x))

x_5_b = 17/2
aux = x_5_b
while (np.absolute(x_5_b-G_x_b(x_5_b))>0.000001):
    aux = x_5_b
    x_5_b = G_x_b(x_5_b)
    fig_5_b = plt.plot((x_5_b,aux),(x_5_b,x_5_b),c='k')
    fig_5_b = plt.plot((x_5_b,x_5_b),(G_x_b(x_5_b),x_5_b),c='k')
    print(x_5_b)


#%%   EXERCICIO 6
# i
"""
Com uma calculadora não programável, encontre a raiz positiva de sin(x) = x/2 usando
os métodos de Newton-Raphson e secantes, com erro  ≤ 10−4
"""

def F_x_6(x):
    return(np.sin(x) - (x/2))

def dF_x_6(x):
    return(np.cos(x) - 0.5)

def ddF_x_6(x):
    return(-np.sen(x))

def g_x_6(x):
    return( x - ( F_x_6(x) / dF_x_6(x) ) )

Erro_6 = 0.000000004

x_n_6 = 10

#CONDICAO:
#Modulo (E_0/2 * (F''(X))/F'(X))<1 OOOUUU MODULO(E_0 * (F''(X)/F'(X))) < 2

while ( np.absolute(x_n_6 - g_x_6(x_n_6)) > Erro_6 ):
    x_n_6 = g_x_6(x_n_6)
    print(x_n_6)

#%%   EXERCICIO 6
# ii

#método das secantes
    
def g_x_6_sec(x_n,x_nm1):
    return(x_n -(  (F_x_6(x_n)*(x_n-x_nm1))/(F_x_6(x_n) - F_x_6(x_nm1))  )  )

Erro_6 = 0.000000004

x_n_6 = 13

aux = 5

while ( np.absolute(x_n_6 - g_x_6_sec(x_n_6,aux)) > Erro_6 ):
    x_n_6_futuro = g_x_6_sec(x_n_6,aux)
    aux = x_n_6
    x_n_6 = x_n_6_futuro
    
    print(x_n_6)

"""
#Colar isso no item ii C do EP1 pra confirmar o metodo das secantes

r_1 = 0.01

r_2 = 0.05

E_r = 1.e-10
#Calculo o erro relativo como critério de parada (r_1 - r_2) / r_2
while(np.absolute((r_1 - r_2)/r_2) >= E_r):
    
    r_m = (r_1+r_2)/2
    
    if(sinal_negativo(F_r(r_1)) == sinal_negativo(F_r(r_m))):
        
        r_1 = r_m
    else:
        r_2 = r_m
    print(r_m)


"""
"""
#TESTE MAIS BASICO
def g_x_ex_3(r_n,r_n_s1): #_s1 é o subtraido por 1
    return(r_n -(  (F_r(r_n)*(r_n-r_n_s1))/(F_r(r_n) - F_r(r_n_s1))  )  )

r_n = 0.005
r_n_s1 = 0.01
E_r = 1.e-13

n = 1 

columns = ['n',
'Rn',
'Rn+1',
'Rn-1',
'F(Rn)',
'Erro Relativo']

tabela_ex_3 = pd.DataFrame(columns = columns)

while (np.absolute((r_n - g_x_ex_3(r_n,r_n_s1))/r_n) >= E_r):
    r_n_a1 = g_x_ex_3(r_n,r_n_s1) #_a1 é adicionado por 1
    r_n_s1 = r_n
    
    erro = (r_n - g_x_ex_3(r_n,r_n_s1))/r_n
    
    df2 = pd.DataFrame([[n,r_n,r_n_a1,r_n_s1,F_r(r_n),
                         erro]],columns = columns)
    
    r_n = r_n_a1
    
    tabela_ex_3 = tabela_ex_3.append(df2, ignore_index=True)
    
    n += 1
#tabela_ex_3 = tabela_ex_3.applymap(${:.5f}".format)   
tabela_ex_3.to_csv('tabela_C_EP1',index=False)

"""

def integral_trapezios(p,f_x,L):
    L = L[1]-L[0]
    for i in range(p):
        n = 2**(i+1)
        h = L/n
        vetor_soma = np.zeros(n+1)
        vetor_soma[0] = f_x(0)
        for j in range(1,len(vetor_soma)-1):
            vetor_soma[j]=2*f_x(h*j)
        vetor_soma[-1] = f_x(n*h)
        soma = np.sum(vetor_soma)
        soma = soma*(h/2)
        print(i+1," ",n," ",soma)

def integral_trapezios(p,f_x,L,sol_an):
    L = L[1]-L[0]
    for i in range(p):
        n = 2**(i+1)
        h = L/n
        vetor_soma = np.zeros(n+1)
        vetor_soma[0] = f_x(0)
        for j in range(1,len(vetor_soma)-1):
            vetor_soma[j]=2*f_x(h*j)
        vetor_soma[-1] = f_x(n*h)
        soma = np.sum(vetor_soma)
        soma = soma*(h/2)
        erro = np.abs(soma - sol_an)
        print(i+1," ",n," ",soma," ",erro)
        
#Agora irei definir o loop
        
p = 17
Im = np.zeros(p)
Sigma = np.zeros(p)
Sigma_m = np.zeros(p)
for i in range(p):
    n = 2**(i+1)
    area_sob_curva = np.zeros(n)
    for k in range(n):
        x = np.zeros(100)
        y = np.zeros(100)
        N_ptos_dentro = 0
        for j in range(100):
            x[j] = linear_cong_gen(1+j,Z_0=10264192.0+k*i)
            #Note que ao modificarmos a seed, garantimos uma combinacao de
            #sequencias aleatorias a cada k
            y[j] = linear_cong_gen(2+j,Z_0=10264192.0-k*i)
            if (((x[j])**4)-y[j]>0):
                N_ptos_dentro+=1
        area_sob_curva[k]=N_ptos_dentro/100
    Im[i] = np.sum(area_sob_curva)/n
    Sigma[i] = np.sqrt(np.sum((area_sob_curva-Im[i])**2)/(n-1))
    Sigma_m[i] = Sigma[i]/np.sqrt(n)
    print("Nt = ", n ,"Im = ", Im[i],"Sigma = ",Sigma[i],"Sigma_m = ", Sigma_m[i])

def linear_cong_gen(n,Z_0=10264192.0):
    Z = Z_0
    a = 1103515245.0
    c = 12345.0
    m = 2147483647.0
    for i in range(n):
        Z = (a*Z + c)%m
    U = Z/m
    return(U)