import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#OBS:No python, a precisão de objetos em float já é dupla pela configuração padrão

#%% -------------------------- EXERCÍCIO A ---------------------------------

"""

Resolva Numericamente a equação:
    
            x³ - cos(x) = 0
            
Usando o método de BISSECÇÃO. Escolher um erro E adequado.
OBS: O ângulo em cos(x) em radianos.

"""

#Obs: o indice _ex1 é referencia ao exercício 1.

#Primeiro irei definir f_x_ex1, que recebe um numero real x e retorna x³ - cos(x)

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

#A escolha dos pontos será dada pelo gráfico da função:

x_ex1 = np.arange(-1,1,0.1)

fig_ex1 = plt.figure(figsize=(8,8),dpi=100)

fig_ex1 = plt.plot(x_ex1,f_x_ex1(x_ex1))

"""
Existem outras raízes?

Notamos que, cos(x) é limitado entre -1 e 1, enquanto que x³ tende à -inf
quando x tende à -inf e quando x tende a +inf, x³ tente à +inf.
Portanto, a raiz dessa função se encontra no intervalo em que |x³| < 1. 
Observamos, pelo gráfico, só haverá uma raíz real no intervalo -1 < x < 1.
"""

x_1_ex1 = 0

x_2_ex1 = 1

E_ex1 = 1.e-5

n = 1

#Fazendo a tabela atraves do pandas
columns = ['n',
'X1',
'X2',
'Xm',
'f(X1)',
'f(X2)',
'f(Xm)',
'Erro Relativo']

tabela_ex_1 = pd.DataFrame(columns = columns)

#Calculo o erro relativo como critério de parada (x_1 - x_2) / x_2

erro = erro_relativo(x_2_ex1,x_1_ex1)
while(np.absolute(erro) >= E_ex1):
    
    x_m_ex1 = (x_1_ex1+x_2_ex1)/2
    
    erro = erro_relativo(x_2_ex1,x_1_ex1)
    
    #A cada iteração, df2 adiciona uma linha, dada por df2
    df2 = pd.DataFrame([[n,x_1_ex1,x_2_ex1,x_m_ex1,f_x_ex1(x_1_ex1),
                    f_x_ex1(x_2_ex1),f_x_ex1(x_m_ex1),erro]],columns = columns)
    
    tabela_ex_1 = tabela_ex_1.append(df2, ignore_index=True)
    n += 1
    
    if(sinal_negativo(f_x_ex1(x_1_ex1)) == sinal_negativo(f_x_ex1(x_m_ex1))):
        
        x_1_ex1 = x_m_ex1
    else:
        x_2_ex1 = x_m_ex1

#Exportando tabela

tabela_ex_1.to_csv('tabela_A_EP1.csv',index=False)


#%% -------------------------- EXERCÍCIO B ---------------------------------

"""
Repetir o item A com método Newton-Raphson
"""

#Obs: o indice _ex2 é referencia ao exercício 2.

#Primeiro irei definir f_x_ex2, que recebe um numero real x e retorna x³ - cos(x)
def f_x_ex2(x):
    return((x**3)-np.cos(x))

#Agora irei definir a função que equivale à primeira derivada de f_x_ex2
def df_x_ex_2(x):
    return(3*(x**2)+np.sin(x))

#Irei definir também a função g_x, pertencente ao método de Newton-Raphson
def g_x_ex_2(x):
    return( x - ( f_x_ex2(x) / df_x_ex_2(x)) ) 

E_ex2 = 1.e-10

x_n_ex_2 = 0.75 #Número suficientemente próximo da raíz

n = 1 

columns = ['n',
'Xn',
'f(Xn)',
'f'+chr(39)+'(Xn)',
'Erro Relativo']

tabela_ex_2 = pd.DataFrame(columns = columns)

erro = erro_relativo(x_n_ex_2,g_x_ex_2(x_n_ex_2))

while ( np.absolute(erro) > E_ex2 ):
    erro = erro_relativo(x_n_ex_2,g_x_ex_2(x_n_ex_2))
    
    #A cada iteração, df2 adiciona uma linha, dada por df2
    df2 = pd.DataFrame([[n,x_n_ex_2,f_x_ex2(x_n_ex_2),df_x_ex_2(x_n_ex_2),
                         erro]],columns = columns)
    
    tabela_ex_2 = tabela_ex_2.append(df2, ignore_index=True)
    x_n_ex_2 = g_x_ex_2(x_n_ex_2)
    n += 1
    
tabela_ex_2.to_csv('tabela_B_EP1.csv',index=False)

#%% -------------------------- EXERCÍCIO C ---------------------------------
#Item i

"""
Faça os gráficos:
   V(r) em função de r
   
       V(r)=- (e²/(4pi e_0 r)) + V_0 exp(-r/r_0)
   
   F(r) em função de r
   
       F(r)= dV(r)/dr = -(e²/4pi*e_0*r²) + (V_0/r_0)*exp(-r/r_0)
       
Do exercício, obtemos o valor da constante:
    
    e²/4pi*e_0 = 14.4 eVA

    V_0 = 1.09*10³ eV
    r_0 = 0.330 A
"""

V_0 = 1.09*10**3

r_0 = 0.330

def V_r(x):
    return(-(14.4/x) + V_0*np.exp(-x/r_0))

def F_r(x):
    return((-14.4/(x**2)) + (V_0/r_0)*np.exp(-x/r_0))

r_ex3i = np.arange(0,10,0.0005)

fig_ex3i = plt.figure(figsize=(8,8),dpi=100)

fig_ex3i = plt.subplot(2,1,1)

fig_ex3i = plt.plot(r_ex3i,V_r(r_ex3i))
fig_ex3i = plt.ylim(-6,0)
fig_ex3i = plt.xlim(1.8,8)
fig_ex3i = plt.grid(True)

fig_ex3i = plt.title('Potencial de Interação')
fig2 = plt.xlabel('r  (Â)')
fig2 = plt.ylabel('V(r)   (eV)')
fig_ex3i = plt.subplot(2,1,2)

fig_ex3i = plt.plot(r_ex3i,F_r(r_ex3i))

fig_ex3i = plt.grid(True)
fig_ex3i = plt.xlim(1.8,8)
fig_ex3i = plt.ylim(-2.5,10)

fig_ex3i = plt.title('Força entre os Ions')
fig2 = plt.xlabel('r  (Â)')
fig2 = plt.ylabel('F(r)   (eV/Â)')
#%% -------------------------- EXERCÍCIO C ---------------------------------
#Item ii

"""
Use o método de secantes e encontre o ponto de equilíbrio r(=req) em A, que é
a solução de F(r) = 0.
"""
#Irei definir a função g_x_ex_3, pertencente ao método das secantes, de forma
#que a função recebe x_n-1 e x_n e devolve x_n+1 e o próprio x_n

def g_x_ex_3(r_n,r_n_s1): #_s1 é o subtraido por 1
    return(r_n -(  (F_r(r_n)*(r_n-r_n_s1))/(F_r(r_n) - F_r(r_n_s1))  ) , r_n )

#Este intervalo dado por r_n e r_n_s1 é suficientemente próximo da raiz de interesse
#conferindo pelo gráfico.

r_n = 3
r_n_s1 = 2.5
E_r = 1.e-13

n = 1

columns = ['n',
'Rn',
'Rn-1',
'F(Rn)',
'Erro Relativo']

tabela_ex_3 = pd.DataFrame(columns = columns)

erro = erro_relativo(r_n,r_n_s1)

while (np.absolute(erro) >= E_r):
    erro = erro_relativo(r_n,r_n_s1)
    
    df2 = pd.DataFrame([[n,r_n,r_n_s1,F_r(r_n),
                        erro]],columns = columns)
        
    r_n,r_n_s1 = g_x_ex_3(r_n,r_n_s1)
    
    n += 1
    
    tabela_ex_3 = tabela_ex_3.append(df2, ignore_index=True)
  
tabela_ex_3.to_csv('tabela_C_EP1.csv',index=False)
