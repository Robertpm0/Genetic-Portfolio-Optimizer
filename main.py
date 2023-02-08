from pandas_datareader import data as pdr
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
yf.pdr_override()

tickrs=["RUN","AMZN","VICI","SNAP","TSLA","AVAV"] # to add user inputs
indices=(len(tickrs))
startDate= dt.datetime(2019,1,1)
endDate=dt.datetime(2021,5,1)
yields = pdr.get_data_fred(['DGS6MO','DGS1','DGS2','DGS10','DGS30'])
y1,y2,y3,y4,y5=(yields.iloc[-1])

#risk free rate/return
#using average of treasury yields
rfr = (y1+y2+y3+y4+y5)/500
print("Risk free rate: ",rfr,'\n')




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


#baseReturn=(np.sum(stockReturns.mean()*[.125,.125,.125,.125,.125,.125,.125,.125]*252))

#algorithm for brute forcing all possible portfolio weightsnp
risks=[]
returns=[]
lbs=[]
sharpe=[]


rots=0
# use this for brute forcing every possible portfolio
def getWeights(n,target,max,arr=[],sum=0):
    global rots
    if len(arr) > n or sum > target:
        return
    if sum == target:
        weights=(arr + [0 for _ in range(n-len(arr))])
        #lbs.append(weights)
        totalReturn=np.sum(stockReturns.mean()*weights*252)
        matCov=stockReturns.cov() *252
        weights = np.array(weights)
        portfolioVariance=np.dot(weights.T,np.dot(matCov,weights))
        portfolioStd=np.sqrt(portfolioVariance)
        print(totalReturn)
        shrpeRatio=(totalReturn-rfr)/ portfolioStd
        risks.append(portfolioStd)
        returns.append(totalReturn)
        lbs.append(weights)
        sharpe.append(shrpeRatio)
        print(weights)
        rots +=1
        print(rots)
        return
    for i in range((max) + 1):
        getWeights(n,target,max,arr+[i/100],sum+i)
        

#much faster but randomly chooses weights

def cFitness(pl,risk,sharpe):
    fitness=[]
    for i in range(fams):
        fit = ((pl[i])/risk[i])*(sharpe[i])
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

def mutation(offspring_crossover,matingLen):

    # Mutation changes a single gene in each offspring randomly.

    for idx in range(offspring_crossover.shape[0]):

        # The random value to be added to the gene.
        randomIdx = np.random.randint(0,numWeights)

        random_value = np.random.uniform(0, 1.0, 1)

        offspring_crossover[idx, randomIdx] = offspring_crossover[idx, randomIdx] + random_value
        
        offspring_crossover[idx]/=offspring_crossover[idx].sum()
    return offspring_crossover
def randomWeights(stocks,folios):
    for i in range(folios):
        a=np.random.random(stocks)
        a/=a.sum()
        weights=a
        totalReturn=np.sum(stockReturns.mean()*weights*252)
        matCov=stockReturns.cov() *252
        portfolioVariance=np.dot(weights.T,np.dot(matCov,weights))
        portfolioStd=np.sqrt(portfolioVariance)
        print(totalReturn)
        shrpeRatio=(totalReturn-rfr)/ portfolioStd
        risks.append(portfolioStd)
        returns.append(totalReturn)
        lbs.append(weights)
        sharpe.append(shrpeRatio)
        print(i)


def Evolve(returns,weights,gens):
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
        oMut=mutation(oCross,matingSize)
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

randomWeights(indices,10000)  
getWeights(4,100,65)

lbs = np.array(lbs)
risks = np.array(risks)
sharpe = np.array(sharpe)
returns = np.array(returns)

print(lbs.shape)
print(risks.shape)
print(sharpe.shape)
print(returns.shape)

aryLen = len(risks)

optimalWeights = lbs[0]
optimalRisk = risks[0]
optimalSharpe = sharpe[0]
optimalReturn = returns[0]
print(optimalReturn,optimalRisk,optimalSharpe,optimalWeights)
for i in range(aryLen-1):
    if risks[i]< optimalRisk and returns[i]>optimalReturn and sharpe[i]> optimalSharpe:
        optimalWeights = lbs[i]
        optimalRisk = risks[i]
        optimalSharpe = sharpe[i]
        optimalReturn = returns[i]
        print("good")
    print(i)
    




print(len(risks))
print(len(returns))
print(len(lbs))
print(len(sharpe))
maxnSharpeIndex=np.argmax(sharpe)
print("Optimal Portfolio (sharpe):",'\n')
print("Sharpe: ",sharpe[maxnSharpeIndex],'\n')
print("Risk: ",risks[maxnSharpeIndex],'\n')
print("Exp. Return: ",returns[maxnSharpeIndex],'\n')
print("Holdings: ",tickrs,'\n')
print("Rec. Allocation: ",lbs[maxnSharpeIndex],'\n')

minRiskIndex = np.argmin(risks)
print("Optimal Portfolio (Min Risk):",'\n')
print("Sharpe: ",sharpe[minRiskIndex],'\n')
print("Risk: ",risks[minRiskIndex],'\n')
print("Exp. Return: ",returns[minRiskIndex],'\n')
print("Holdings: ",tickrs,'\n')
print("Rec. Allocation: ",lbs[minRiskIndex],'\n')
maxRiskIndex = np.argmax(risks)
print("Optimal Portfolio (Max Risk):",'\n')
print("Sharpe: ",sharpe[maxRiskIndex],'\n')
print("Risk: ",risks[maxRiskIndex],'\n')
print("Exp. Return: ",returns[maxRiskIndex],'\n')
print("Holdings: ",tickrs,'\n')
print("Rec. Allocation: ",lbs[maxRiskIndex],'\n')
print("Optimal Portfolio:",'\n')
print("Sharpe: ",optimalSharpe,'\n')
print("Risk: ",optimalRisk,'\n')
print("Exp. Return: ",optimalReturn,'\n')
print("Holdings: ",tickrs,'\n')
print("Rec. Allocation: ",optimalWeights,'\n')
#print("Base Return: ",baseReturn-rfr)


plt.scatter(risks,returns,c=sharpe)
plt.xlabel("Volatility")
plt.ylabel("Returns")
plt.colorbar(label="Sharpe Ratio")
plt.savefig("opti3.png")
plt.show()

 
