import numpy as np
import random
import epsilonGreedy as eG
from collections import defaultdict
def chooseOptimalArm(successes, failures, totalArms):
    betaSamples=np.zeros(totalArms)
    for i in range(totalArms):
        betaSamples[i]=np.random.beta(1+successes[i],1+failures[i])
    listOfMaxArm = np.where(betaSamples == betaSamples.max())[0]
    return np.random.choice(listOfMaxArm)

def thompsonSampling(arms, randomSeed, horizon):
    np.random.seed(randomSeed)
    random.seed(randomSeed)
    maxTrueMean = np.array(arms).max()
    successes = np.zeros(len(arms))
    failures = np.zeros(len(arms))
    totalEmpericalReward=0
    listOfArmPulls=[]
    countPull=defaultdict(int)
    for t in range(horizon):
        arm = chooseOptimalArm(successes, failures, len(arms))
        countPull[arm]+=1
        listOfArmPulls.append(arm)
        tempReward = eG.generateReward2(arm, arms)
        successes[arm]+=tempReward
        failures[arm] +=1-tempReward
        totalEmpericalReward += tempReward
    regret = horizon * maxTrueMean - totalEmpericalReward
    return regret

'''def calculatePriorAlphaBeta(hint):
    mean=hint.mean()
    variance=hint.var()
    alpha=mean*mean*(((1-mean)/variance)-1/mean)
    beta=alpha*(1/mean-1)
    return (alpha, beta)
def chooseOptimalArmWithHint2(totalArms, hint, successes, failures):
    alpha, beta = calculatePriorAlphaBeta(np.array(hint))
    betaSamples = np.zeros(totalArms)
    for i in range(totalArms):
        betaSamples[i] = np.random.beta(alpha + successes[i], beta + failures[i])
    listOfMaxArm = np.where(betaSamples == betaSamples.max())[0]
    return np.random.choice(listOfMaxArm)'''
def chooseOptimalArmWithHint(beliefs, totalArms, hint, successes, failures):

    possibleOptimal = np.where(beliefs == beliefs.max())[0]
    betaSamples = np.zeros(totalArms)
    for i in range(totalArms):
        betaSamples[i] = np.random.beta(1 + successes[i], 1 + failures[i])
    maxBeta= betaSamples.max()
    listOfMaxArm=[]
    if(2*hint[totalArms-1]-hint[totalArms-2]>=maxBeta and maxBeta>=hint[totalArms-2]):
        x = np.where(betaSamples > hint[totalArms-2])[0]
        for a in x:
            if(a in possibleOptimal):
                listOfMaxArm.append(a)
    if(len(listOfMaxArm)==0):
        listOfMaxArm = np.where(betaSamples == maxBeta)[0]
    return np.random.choice(listOfMaxArm)

def thompsonSamplingWithHint(arms, randomSeed, horizon, hint):
    random.seed(randomSeed)
    np.random.seed(randomSeed)
    successes = np.zeros(len(arms))
    failures = np.zeros(len(arms))
    maxTrueMean = np.array(arms).max()
    beliefs = np.zeros(len(arms))
    beliefs.fill(hint[len(arms)-1])
    countPull = defaultdict(int)
    totalEmpericalReward=0
    for t in range(horizon):
        #arm=chooseOptimalArmWithHint2(len(arms), hint, successes, failures)
        arm = chooseOptimalArmWithHint(beliefs, len(arms), hint, successes, failures)
        countPull[arm]+=1
        tempReward = eG.generateReward2(arm, arms)
        successes[arm] += tempReward
        failures[arm] += 1 - tempReward
        beliefs[arm]=(successes[arm] + 1)/(successes[arm] + failures[arm]+2)
        totalEmpericalReward += tempReward
    regret = horizon * maxTrueMean - totalEmpericalReward
    return regret
