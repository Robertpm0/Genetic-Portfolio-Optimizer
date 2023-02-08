from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
yf.pdr_override()

import tkinter as tk
import customtkinter

tickrs=["SPY","VICI","SOXL","SOXS","BBH"] # to add user inputs
indices=(len(tickrs))
startDate= dt.datetime(2019,1,1)
endDate=dt.datetime(2021,5,1)
yields = pdr.get_data_fred(['DGS6MO','DGS1','DGS2','DGS10','DGS30'])
y1,y2,y3,y4,y5=(yields.iloc[-1])

#risk free rate/return
#using average of treasury yields
rfr = (y1+y2+y3+y4+y5)/500
print("Risk free rate: ",rfr,'\n')



#loading returns per indice
stockReturns = pd.DataFrame()

for ticker in tickrs:
    daat=pdr.get_data_yahoo(ticker,startDate,endDate)
    daata = pd.DataFrame(daat)
    daata[ticker]=daata['Adj Close'].pct_change()
    if stockReturns.empty:
        stockReturns=daata[[ticker]]
    else:
        stockReturns=stockReturns.join(daata[[ticker]],how='outer')

tReturns = np.array(stockReturns.mean()*252)

numWeights = len(tickrs)
fams = 8
#matingSize = numWeights-2
matingSize = 4
popSize =(fams,numWeights)
initialWeights = np.random.random(popSize)
#norming each population to a sum of 1
for i in range(fams):
    initialWeights[i,:]=initialWeights[i,:]/initialWeights[i,:].sum()
#initialWeights/=initialWeights.sum
print(initialWeights)






#components of genetic algorithm.
#inspired by Ahmed Gad on Medium

#fitness function
# f(x) = (PnL / Risks) * Sharpe Ratio
# could be improved but typically finds
# well balanced portfolio
def cFitness(pl,risk,sharpe):
    fitness=[]
    for i in range(fams):
        fit = ((pl[i])/risk[i])*np.abs((sharpe[i]))
        fitness.append(fit)
    return fitness

def selectMatingPool(weights, fitness, matingSize):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((matingSize, weights.shape[1]))
    for parent_num in range(matingSize):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = weights[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
     # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)
 
    for k in range(offspring_size[0]):
         # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
         # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
         # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
         # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):

        # The random value to be added to the gene.
        randomIdx = np.random.randint(0,numWeights)

        random_value = np.random.uniform(0, 1.0, 1)

        offspring_crossover[idx, randomIdx] = offspring_crossover[idx, randomIdx] + random_value
        
        offspring_crossover[idx]/=offspring_crossover[idx].sum()
    return offspring_crossover

#main driver of genetic algorithm
def Evolve(returns,weights,gens):
    #these are used for plotting all portfolios
    pls=[]
    riskss=[]
    sharpes=[]
    for generation in range(gens):
        pl=[]
        risks=[]
        sharpe=[]
        matCov = returns.cov()*252
        for i in range(fams):
            pnl=np.sum(returns.mean()*252*weights[i,:])
            pv=np.dot(weights[i,:].T,np.dot(matCov,weights[i,:]))
            pstd=np.sqrt(pv)
            shrpe = (pnl-rfr)/pstd
            pl.append(pnl)
            pls.append(pnl)
            risks.append(pstd)
            riskss.append(pstd)
            sharpe.append(shrpe)
            sharpes.append(shrpe)
        print("Gen Number", generation)
        finess = cFitness(pl,risks,sharpe)
        parents = selectMatingPool(initialWeights,finess,matingSize)
        oCross =crossover(parents,offspring_size=(popSize[0]-parents.shape[0],numWeights))
        oMut=mutation(oCross)
        initialWeights[0:parents.shape[0], :] = parents
        initialWeights[parents.shape[0]:, :] = oMut
    return pl,risks,sharpe,pls,riskss,sharpes
pl,risks,sharpe,pls,riskss,sharpes=Evolve(stockReturns,initialWeights,gens=200)
print(pl,risks,sharpe)
lastFit = cFitness(pl,risks,sharpe)
finalidx = np.where(lastFit==np.max(lastFit))

print("BEST PORTFOLIO:",'\n')
print(initialWeights[finalidx,:],'\n')
pl = np.array(pl)
risks = np.array(risks)
sharpe = np.array(sharpe)
print(pl)
print("Return: ", pl[finalidx],'\n')
print("Risk: ",risks[finalidx],'\n')
print("Sharpe Ratio: ",sharpe[finalidx])


plt.scatter(riskss,pls,c=sharpes)
plt.xlabel("Volatility")
plt.ylabel("Returns")
plt.colorbar(label="Sharpe Ratio")
plt.savefig("opti3.png")
plt.show()
