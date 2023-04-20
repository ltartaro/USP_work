import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Para todos os exercícios, basta declarar uma matriz que represente um sistema
de equações lineares L.I, nx(n+1), no lugar de matriz_prob. A matriz deve ser
um objetito array do pacote Numpy.
"""

#%% -------------------------- EXERCÍCIO B ---------------------------------

"""
Resolver o sistema

[0   5 -1] [I1 ] = [4 ]
[13  0  1] [I2 ] = [15]
[1  -1 -1] [I3 ] = [0 ]

-Usando o metodo de Eliminacao de Gauss usando pivotamento parcial.
-IMPRIMIR as matrizes intermediarias ate alcancar a matriz triangular superior.
-Programa capaz de resolver sistemas de n equacoes
"""

#Irei definir uma função que determina o valor maximo em um vetor e sua posicao
def max_val(vetor):
    val_max = vetor[0]
    posicao = 0
    for i in range(1,len(vetor)):
       if(vetor[i]>val_max):
           val_max=vetor[i]
           posicao = i
    return(val_max,posicao)

#Nao esquecer que as matrizes sao da forma a[i][j]

#Primeiro irei criar um algoritmo para pivotamento.

#TODAS AS ROTINAS RECEBEM UMA MATRIZ GENERICA nx(n+1), isto é, 
#a matriz de coeficientes junto do vetor das constantes.
#Basicamente, ela é capaz de resolver sistemas L.I

def pivotamento_parcial(matriz):
    print("Matriz Inicial\n")
    print(matriz,"\n")
    print("Pivotamento Parcial\n")
    for i in range(np.shape(matriz)[0]): #so percorre linhas porque estamos interessados apenas no elemento da diagonal
        valor_maximo,pos=max_val(np.abs(matriz[:,i])) # EM MODULO
        if(i<pos): #Evita desorganizar linhas ja pivotadas
            aux =np.copy(matriz[i]) #importante fazer a copia para que aux nao #sofra modificacao
            matriz[i] = matriz[pos]
            matriz[pos] = aux
            print(matriz,"\n")
    return(matriz)

#Agora vou criar algoritmo para gerar matriz triangular superior:

def triangular_superior(matriz):
    matriz = pivotamento_parcial(matriz)
    print("Matriz Triangular superior\n")
    for i in range(np.shape(matriz)[0]-1): #aqui percorremos as colunas
        for j in range(i,np.shape(matriz)[0]-1): #Aqui percorremos as linhas
            aux=matriz[j+1][i]/matriz[i][i] #Aqui calculamos o coeficiente aux
            matriz[j+1] = matriz[j+1]-matriz[i]*aux #Aqui somamos a linha i na linha j e substituimos
            print(matriz,"\n")
    return(matriz)
            
#Vou criar o algorítimo para resolver a equação fazendo as substituicoes
def resol_matr_trian(matriz_t_s):
    linhas = np.shape(matriz_t_s)[0]
    vet_sol = np.zeros(linhas)
    for i in range(1,linhas+1):          
        aux=matriz_t_s[linhas-i][linhas] #Note que aqui, estamos percorrendo o #vetor para tras
        for j in range(1,i+1): #A cada iteracao, incrementamos o valor de aux  #com as solucoes ja obtidas
            aux += -matriz_t_s[linhas-i][linhas-j]*vet_sol[linhas-j] 
        vet_sol[linhas-i] = aux/matriz_t_s[linhas-i][linhas-i] 
    return(vet_sol)

#%%
#Definindo a matriz do problema nx(n+1)
matriz_prob = np.array([[0., 5., -1., 4.],
                       [13., 0., 1., 15.],
                       [1., -1., -1., 0.]])

triangular_superior(matriz_prob)
solucao = resol_matr_trian(matriz_prob)
print("Solução do sistema\n")
print(solucao.reshape(np.shape(solucao)[0],1),"\n")

#%% -------------------------- EXERCÍCIO C ---------------------------------
"""
Utilizar o metodo de Jacobi para a matriz abaixo.
-Criterio de parada: max|xi(k+1) - xi(k)| < E, para i = 1, ..., n
- E = 10^(-4)
- Imprimir a tabela contendo k (numero da iteracao); valores de I1, I2, I3; Erro = E
- Programa capaz resolver sistemas de n equacoes
"""

#DEFININDO A MATRIZ QUE REPRESENTA O SISTEMA DADA POR matriz_prob nx(n+1)

matriz_prob = np.array([[13., 0., 1., 15.],
                        [0., 5., -1., 4.],
                        [1., -1., -1., 0.]])


solucao = np.ones(np.shape(matriz_prob)[0])
E=1 #Valor arbitrario maior que o erro para iniciar o loop, nao sera tabelado

k=1

#Gerando a tabela do exercicio C
columns = ['k',
'I1',
'I2',
'I3',
'Erro']
ccc

print("Solucao pelo Metodo de Jacobi \n")
print("K     Solução = [ I1           I2           I3  ]    Erro  =   E")

"""
Este codigo calcula a solucao do sistema percorrendo as linhas e as colunas da
matriz, isolando cada icognita de cada linha. Note que aqui, existe
"""
while(E>1e-4):
    solucao_n_1 = np.copy(matriz_prob[:,-1])
    for i in range(np.shape(solucao)[0]): #aqui percorremos linha a linha
        for j in [x for x in range(np.shape(solucao)[0]) if x != i]: #Aqui percorremos as colunas, exceto a diagonal
            solucao_n_1[i] += -np.copy(matriz_prob[i][j])*solucao[j] #Nao modificamos o vetor solucao
        solucao_n_1[i] = solucao_n_1[i]/np.copy(matriz_prob[i][i]) 
    E=max_val(np.abs(solucao - solucao_n_1));E = E[0] #O programa max_val gera uma tupla, com valor max e posicao no vetor
    solucao = solucao_n_1
    print("K =",k,"Solução = ",solucao," Erro = ", E)
    df1 = pd.DataFrame([[k,solucao[0],solucao[1],solucao[2],E]],columns = columns)
    tabela_ex_C = tabela_ex_C.append(df1, ignore_index=True)
    k +=1
    
#Exportando tabela

tabela_ex_C.to_csv('tabela_ex_C_EP2.csv',index=False)

#%% -------------------------- EXERCÍCIO D ---------------------------------
"""
Utilizar o metodo de Gauss-Seidel para a matriz abaixo.
-Criterio de parada: max|xi(k+1) - xi(k)| < E, para i = 1, ..., n
- E = 10^(-4)
- Imprimir a tabela contendo k (numero da iteracao); valores de I1, I2, I3; Erro = E
- Programa capaz resolver sistemas de n equacoes
"""

#DEFININDO A MATRIZ QUE REPRESENTA O SISTEMA DADA POR matriz_prob nx(n+1)

matriz_prob = np.array([[13., 0., 1., 15.],
                        [0., 5., -1., 4.],
                        [1., -1., -1., 0.]])

#Gerando a tabela do exercicio D
tabela_ex_D = pd.DataFrame(columns = columns)
print("\n Solucao pelo Metodo de Gauss-Seidel \n")
print("K     Solução = [ I1           I2           I3  ]    Erro  =   E")
"""
A diferença deste metodo para o anterior será aproveitar a solução obtida para
a variável na própria iteração. Então ao invés de calcular a solucao para uma
variável x[i] utilizando um vetor solucao fixo "solucao" e escrevendo um vetor novo
solucao_n_1, irei simplesmente utilizar um vetor "solucao" e escrever cada valor
da variavel no proprio vetor.

Note que a variavel "solucao_ante" é utilizada apenas para calcular o erro para
cada iteração.
"""

"""
Ao compararmos as tabelas, fica clara a diferenca na convergencia do método
de Gauss-Seidel quando comparado ao de Jacobi
"""

solucao = np.ones(np.shape(matriz_prob)[0])
E=1 #Valor arbitrário maior que o erro para iniciar o loop, nao sera 
k=1
while(E>=1e-4):
    for i in range(np.shape(solucao)[0]): #aqui percorremos linha a linha
        solucao_ante = np.copy(solucao) # este vetor serve apenas para calcular # o erro
        solucao[i] = np.copy(matriz_prob[i][-1])
        for j in [x for x in range(np.shape(solucao)[0]) if x != i]:#Aqui percorremos as colunas, exceto a diagonal
            solucao[i] += -np.copy(matriz_prob[i][j])*solucao[j]#Agora modificamos a solucao a cada iteracao
        solucao[i] = solucao[i]/np.copy(matriz_prob[i][i])
    E=max_val(np.abs(solucao - solucao_ante));E = E[0] #O programa max_val gera #uma tupla, com valor max e posicao no vetor
    print("K =",k,"Solução = ",solucao," Erro = ", E)
    df2 = pd.DataFrame([[k,solucao[0],solucao[1],solucao[2],E]],columns = columns)
    tabela_ex_D = tabela_ex_D.append(df2, ignore_index=True)
    k +=1

#Exportando tabela

tabela_ex_D.to_csv('tabela_ex_D_EP2.csv',index=False)