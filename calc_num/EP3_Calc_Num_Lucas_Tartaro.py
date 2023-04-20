import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% -------------------------- EXERCÍCIO 1 ---------------------------------

"""
Calcular a integral, no intervalo L=[0,1], da função:
    f(x)= 7 - 5x^4
Cuja solucao analitica para o intervalo L=[0,1] é 6.
 
"""

#ITEM A

#irei definir a funcao f(x) que recebe um vetor x=(x1,x2...,xn) e devolve um
#vetor onde cada coordenada representa o f(x_i).
"""
Note que tanto em "f_x" quanto em "integral_trapezios" foi definida prec
para poder modificar entre float com precisao simples e precisao dupla.
"""

def f_x(x,prec=np.float64):
    return(prec(7)-prec(5)*(prec(x)**prec(4)))

"""
Agora irei definir uma funcao que calcula numericamente uma integral usando
o metodo dos trapezios.

Como entradas ela recebe:
    p: de forma que N=2^p é o numero de subdivisoes do intervalo L
    f(x): a funcao a ser integrada
    L: o intervalo da integracao dado por um par ordenado
"""

#este programa define sol_an pois é possivel utiliza-lo sem saber o valor da
#integral analiticamente.
def integral_trapezios(p,f_x,L,sol_an=np.nan,prec=np.float64):
    columns = ['p',
               'N',
               'Inum',
               'Erro']
    tab_int_trap = pd.DataFrame(columns = columns)
    L = L[1]-L[0] # Calculando intervalo
    L = prec(L)
    for i in range(p): #p o numero de iteracoes
        n = 2**(i+1) #i+1 porque i comeca em zero e termina em 24
        h = L/n
        #np.arange gera um vetor com cada xi. assim aplicamos f_x em cada
        #coordenada do vetor gerado
        vetor_soma = f_x(np.arange(0,L+h,h),prec=prec)
        #o seguinte slice seleciona todos os mentros x_2 até x_(n-1)
        vetor_soma[1:-1] = prec(2)*prec(vetor_soma[1:-1])
        soma = np.sum(vetor_soma,dtype=prec)
        soma = prec(soma)*prec(prec(h)/prec(2))
        if(sol_an != np.nan):
            erro = np.abs(prec(soma) - prec(sol_an),dtype=prec)
        df = pd.DataFrame([[i+1,n,soma,erro]],columns=columns,dtype=prec)
        tab_int_trap = tab_int_trap.append(df,ignore_index=True)
    return(tab_int_trap)

#Gerando a tabela para precisao simples
Tabela_prec_simp=integral_trapezios(25,f_x,[0,1],6,prec=np.float32)
print(" Tabela com float prec simples")
print(Tabela_prec_simp)

#O np.where ira identificar onde o erro se iguala a zero e irá eliminar o valor
#substutindo-o por np.nan.
Erro_prec_simp=np.where(Tabela_prec_simp.Erro!=0,Tabela_prec_simp.Erro,np.nan)

#Gerando a tabela para precisao dupla
Tabela_prec_dup=integral_trapezios(25,f_x,[0,1],6,prec=np.float64)
print("\n Tabela com float prec dupla")
print(Tabela_prec_dup)
Erro_prec_dup=np.where(Tabela_prec_dup.Erro!=0,Tabela_prec_dup.Erro,np.nan)
#%%
Fig_item_b=plt.figure(figsize=(8,5),dpi=100)
Fig_item_b=plt.scatter(np.arange(1,26,1),
                       np.log10(Erro_prec_simp),label = "Prec. Simples",s=55)
Fig_item_b=plt.scatter(np.arange(1,26,1),
                       np.log10(Erro_prec_dup),label = "Prec. Dupla",marker='d')
Fig_item_b = plt.legend()
Fig_item_b = plt.title('Estudo Precisão Simples e Dupla')
Fig_item_b = plt.xlabel('p')
Fig_item_b = plt.ylabel('log(|erro|)')

#%% -------------------------- EXERCÍCIO 2 ---------------------------------
"""
Utilizar o método de Simpson para calcular a integral no intervalo [0,pi/2]

T=4*np.sqrt(l/g)*integral(1/np.sqrt(1-(k**2)*(np.sin(x)))dx

    Onde k=np.sin(theta_0/2)
"""

"""
Primeira parte: Fazer a tabela com 10 iteracoes para theta_0 e T, com
theta_0 variando no intervalo [0,np.pi/2] 

Ou seja, o invervalo será 

np.linspace(0,np.pi/2,10)

"""

"""
Definindo a funcao T em funcao do angulo x, que depende de parametros inidicias
tais como
    l: comprimento do pendulo. Utilizarei o valor arbitrario de l=1
    theta_0: angulo inicial
T_x recebe um vetor e devolve outro vetor com os valores dos integrandos
"""

def T_x(x,theta_0=0,l=1):
    k=np.sin(theta_0/2)
    g=9.8
    return((4*np.sqrt(l/g)) * (1/(np.sqrt(1-(k**2)*((np.sin(x))**2)))))

"""
Irei definir minha funcao de integracao por simpson  que recebe:
    f_x: a ser integrada
    L: O intervalo de integracao [x1,x2]
    p: de forma que N=2^p é o numero de subdivisoes do intervalo L
    param: parametros da funcao, como o valor de theta_0
"""

def integral_simpson(p,f_x,L,param):
    n = 2**p
    L = L[1]-L[0]
    h = L/n
    #np.arange ira subdividir meu intervalos com tamanho h calculado
    vetor_soma = f_x(np.arange(0,L+h,h),param)
    #O seguinte slide seleciona os pares do segundo ao penultimo valor
    vetor_soma[1:-1:2] = 4*vetor_soma[1:-1:2]
    #O seguinte slide seleciona os impares do terceiro ao anti-penultimo valor
    vetor_soma[2:-2:2] = 2*vetor_soma[2:-2:2]
    vetor_soma = (h/3)*vetor_soma
    soma = np.sum(vetor_soma)
    return(soma)

"""
Agora basta construir um loop para gerar a tabela
"""

columns = ['Theta_0',
           'T']
tab_int_simpson = pd.DataFrame(columns = columns)

Num_div = 12

for i in np.linspace(0,np.pi,10,endpoint=False):
    df = pd.DataFrame([[i,integral_simpson(Num_div,T_x,[0,np.pi/2],i)]]
                      ,columns=columns)
    tab_int_simpson = tab_int_simpson.append(df,ignore_index=True)

print("\n Tabela theta_0 por T")
print(tab_int_simpson)
#Item Grafico
    
def T_galileu(l=1):
    g=9.8
    return(2*np.pi*np.sqrt(l/g))

Num_div = 12

theta_0 = np.linspace(0,np.pi,1000,endpoint=False)

valores_T = np.zeros(len(theta_0))

k=0

for i in theta_0:
    T = integral_simpson(Num_div,T_x,[0,np.pi/2],i)
    T = T/(T_galileu())
    valores_T[k] = T
    k+=1
    
Fig_item_2 = plt.figure(figsize=(8,5),dpi=100)
Fig_item_2 = plt.plot(theta_0,valores_T)
Fig_item_2 = plt.xlabel('Theta_0')
Fig_item_2 = plt.ylabel('T/(T_Galileu)')


#%% -------------------------- EXERCÍCIO 3 ---------------------------------

#Item A
"""
Irei definir uma funcao que recebe um numero real N e devolve um vetor de numeros
aleatorios de tamanho N.
"""

def linear_cong_gen_vet(n,passo=1,Z_0=10264192.0):
    Z = Z_0
    vet_aleat = np.zeros(n)
    a = 1103515245.0
    c = 12345.0
    m = 2147483647.0
    for i in range(n):
        Z = (a*Z + c)%m
        U = Z/m
        vet_aleat[i] = U
    return(vet_aleat)

#Item B
    
"""
Para determinar se um ponto está sob a curva, basta que, dadas as coordenadas
(x,y), que y < x^4, i.e, x^4 - y < 0
"""

val_aleat = linear_cong_gen_vet(200)

#tomarei metade dos 200 numeros aleatorios para x e a outra metade para y
x = val_aleat[:100]

y = val_aleat[100:]

dif = np.copy((x**4) - y)

#Observando a diferenca entre x^4 e y, basta contar quantos valores sao maiores
#que zero com np.where. O tamnho deste vetor é a contagem dos valores procurados.

N_ptos_dentro = len(np.where(dif>0)[0])

area_sob_curva = N_ptos_dentro/100

print("\n Item B: Numero de pontos dentro da curva para uma tentativa:")
print(area_sob_curva)

#Item C
"""
Agora irei aplicar a o mesmo algoritmo para um numero maior de tentativas
"""

p = 17
Im = np.zeros(p)
Sigma = np.zeros(p)
Sigma_m = np.zeros(p)

#reservando um vetor de numeros aleatorios:
tam_vet_ale = 0
n=0
"""
A ideia para essa parte será reservar um vetor com todos o numeros aleatorios
necessarios para todas as tentativas, a fim de aumentar bastante a velocidade 
de execussao do programa.

Assim, irei definir o numero total de pts somando 
2*200 + 4*200 + ...+ 131072*200:
"""
for i in range(p+1):
    n = 2**(i+1)
    tam_vet_ale += n
"""
Agora, basta gerar meu vetor de numeros aleatorios com a sequencia de tamanho
"tam_vet_ale"
"""
vet_ale = linear_cong_gen_vet(tam_vet_ale*200)

"""
Agora basta separar cada 100 ptos nos numeros de tentativas, isso é, 400 numeros
aleatorios gerando 200 ptos separados em conjuntos de 100 para calcular a inte-
gral para a primeira iteração, 800 numeros aleatorios para a segunda e assim 
sucessivamente.

No entanto, já irei calcular a diferenca entre x e f(x) para saber se  o ponto
se encontra acima ou abaixo da curva y = x^4.
Como assumimos que os numeros sao aleatorios, irei didivir meu vetor pela metade
onde a primeira etade sera x e a segunda y. O sinal da diferenca (se positivo)
me dira se o ponto se encontra abaixo ou acima da curva.
"""

columns = ['Nt',
           'Im',
           'Sigma',
           'Sigma_m']

tab_int_montcarl = pd.DataFrame(columns = columns)

#vou tomar metade dos numeros aleatorios para x, outra metade para y:
x = np.copy(vet_ale[0:(np.shape(vet_ale)[0]//2)])
y = np.copy(vet_ale[(np.shape(vet_ale)[0]//2):])
#Testando quais pontos estao dentro ou fora da curva, basta a subtracao
#de x**4 - f(x)
dif = np.copy((x**4) - y)
#Agora basta separar cada intervalo de jogadas
for i in range(p):
    n = 2**(i+1)
    area_sob_curva = np.zeros(n)
    for k in range(n):
        #Desta forma eu separo meu vetor de diferencas de 100 em 100 pontos
        res=np.copy(dif[k*100:100+(100*k)])
        N_ptos_dentro = len(np.where(res>0)[0])
        area_sob_curva[k] = N_ptos_dentro/100
    Im[i] = np.sum(area_sob_curva)/n
    Sigma[i] = np.sqrt(np.sum((area_sob_curva-Im[i])**2)/(n-1))
    Sigma_m[i] = Sigma[i]/np.sqrt(n)
    df = pd.DataFrame([[n,Im[i],Sigma[i],Sigma_m[i]]]
                      ,columns=columns)
    tab_int_montcarl = tab_int_montcarl.append(df,ignore_index=True)

print("\n Tabela Integral de Monte Carlo")
print(tab_int_montcarl)

Tabela_prec_simp.to_csv('tabela_ex_1_simp_EP3.csv',index=False)
Tabela_prec_dup.to_csv('tabela_ex_1_dup_EP3.csv',index=False)
tab_int_simpson.to_csv('tabela_ex_2_EP3.csv',index=False)
tab_int_montcarl.to_csv('tabela_ex_3_EP.csv',index=False)