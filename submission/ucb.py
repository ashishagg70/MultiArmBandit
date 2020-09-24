import epsilonGreedy as eG
import numpy as np
import time
import random
from collections import defaultdict
def chooseOptimalArmUCB(empericalMeanArms, countPullArms, t):
    ucbVals = np.zeros(len(empericalMeanArms))
    for i in range(len(ucbVals)):
        ucbVals[i] = empericalMeanArms[i] + np.sqrt(2*(np.log(t))/ countPullArms[i])
    listOfMaxArm=np.where(ucbVals == ucbVals.max())[0]
    return np.random.choice(listOfMaxArm)

def ucb(arms, randomSeed, horizon):
    np.random.seed(randomSeed)
    random.seed(randomSeed)
    maxTrueMean = np.array(arms).max()
    empericalMeanArms = np.zeros(len(arms))
    countPullArms = np.zeros(len(arms))
    totalRewardArms = np.zeros(len(arms))
    countPull = defaultdict(int)
    totalEmpericalReward = eG.sampleEachArmOnceInitially(arms, countPullArms, totalRewardArms, empericalMeanArms,horizon)
    for t in range(len(arms), horizon):
        arm = chooseOptimalArmUCB(empericalMeanArms, countPullArms, t)
        countPull[arm] += 1
        tempReward = eG.generateReward2(arm, arms);
        countPullArms[arm] += 1
        totalRewardArms[arm] += tempReward
        empericalMeanArms[arm] = totalRewardArms[arm] / countPullArms[arm]

        totalEmpericalReward += tempReward;
    regret = horizon * maxTrueMean - totalEmpericalReward
    return regret

def KLV(x, y):
    return  x * np.log(x / y)+(1-x)*np.log((1-x)/(1-y))

def vectorizedGetMax(p_a, u_a, limit):
    error=1e-5
    error2 = 1e-10
    l=np.array(p_a[:])
    r=np.ones(len(p_a))-error2
    length=len(u_a)
    p_a[p_a==1]-=error2
    p_a[p_a == 0] += error2
    while(1):
        mid=(l+r)/2
        KLLHS=u_a*KLV(p_a,mid)
        if(np.sum(np.logical_or(r-l<error, KLLHS==limit))==length):
            return mid
        l[KLLHS<limit]=mid[KLLHS<limit]
        r[KLLHS>limit]=mid[KLLHS>limit]


def chooseOptimalArmUCBKL(empericalMeanArms, countPullArms, t, klucbVals, prevpulledArm):
    c=3
    limit=np.log(t)+c*np.log(np.log(t))
    klucbVals=vectorizedGetMax(np.array(empericalMeanArms), np.array(countPullArms), limit)
    listOfMaxArm=np.where(klucbVals == klucbVals.max())[0]
    return np.random.choice(listOfMaxArm)

def kl_ucb(arms, randomSeed, horizon):
    np.random.seed(randomSeed)
    random.seed(randomSeed)
    maxTrueMean = np.array(arms).max()
    empericalMeanArms = np.zeros(len(arms))
    countPullArms = np.zeros(len(arms))
    totalRewardArms = np.zeros(len(arms))
    klucbVals = np.zeros(len(arms))
    totalEmpericalReward = eG.sampleEachArmOnceInitially(arms, countPullArms, totalRewardArms, empericalMeanArms, horizon)
    prevpulledArm=0
    for t in range(len(arms), horizon):
        arm = chooseOptimalArmUCBKL(empericalMeanArms, countPullArms, t, klucbVals, prevpulledArm)
        prevpulledArm=arm
        tempReward = eG.generateReward2(arm, arms)
        countPullArms[arm] += 1
        totalRewardArms[arm] += tempReward
        empericalMeanArms[arm] = totalRewardArms[arm] / countPullArms[arm]
        totalEmpericalReward += tempReward
    regret = horizon * maxTrueMean - totalEmpericalReward
    return regret

'''def getMaxQ(l,r, u_a, p_a, limit):
    error=1e-5
    #print(KL(p_a, (l+r)*0.5))
    while(1):
        mid=l+(r-l)/2
        #print(mid)
        if(r-l<error or u_a*KL(p_a, mid)==limit):
            return mid
        elif(u_a*KL(p_a, mid)<limit):
            l=mid
        elif(u_a*KL(p_a, mid)>limit):
            r=mid
            #print('r/ l= %f, r=%f' % (l, r))
            
            
def KL(x, y):
    error=1e-16
    if(x>=0 and x<=error):
        return np.log(1/(1-y))
    elif(x<=1 and x>=1-error):
        return np.log(1/y)
    return  x * np.log(x / y)+(1-x)*np.log((1-x)/(1-y))'''
