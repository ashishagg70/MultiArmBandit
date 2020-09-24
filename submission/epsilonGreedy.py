import numpy as np
import random
from collections import defaultdict


def selectArm(epsilon, totalNoOfArms, empericalMeanArms):
    arm=np.random.randint(totalNoOfArms)
    if(np.random.uniform(0,1)>epsilon):
        arm=chooseOptimalArm(empericalMeanArms)
    return arm

def generateReward2(arm, arms):
    return random.choices([0, 1], weights=[1-arms[arm], arms[arm]], k=1)[0]

def chooseOptimalArm(empericalMeanArms):
    listOfMaxArm=np.where(empericalMeanArms == empericalMeanArms.max())[0]
    return np.random.choice(listOfMaxArm)

def generateRewared(arm, arms):
    return np.random.binomial(1,arms[arm],1)[0] #no bw {0,1}, probability of 1, 1 exp
def sampleEachArmOnceInitially(arms, countPullArms, totalRewardArms, empericalMeanArms, horizon):
    temptotalEmpericalReward=0
    for arm in range(len(arms)):
        if(arm+1>horizon):
            break
        tempReward = generateRewared(arm, arms)
        #tempReward = generateReward2(arm, arms);
        countPullArms[arm] += 1
        totalRewardArms[arm] += tempReward
        empericalMeanArms[arm] = totalRewardArms[arm] / countPullArms[arm]
        temptotalEmpericalReward += tempReward
    return temptotalEmpericalReward
def epsilonGreedyAlgo(arms, randomSeed, epsilon, horizon):
    np.random.seed(randomSeed)
    random.seed(randomSeed)
    maxTrueMean = np.array(arms).max()
    empericalMeanArms = np.zeros(len(arms))
    countPullArms=np.zeros(len(arms))
    totalRewardArms=np.zeros(len(arms))
    countPull = defaultdict(int)
    totalEmpericalReward=sampleEachArmOnceInitially(arms, countPullArms, totalRewardArms, empericalMeanArms, horizon)
    for t in range(len(arms),horizon):
        arm = selectArm(epsilon, len(arms), empericalMeanArms)
        countPull[arm] += 1
        tempReward = generateReward2(arm, arms)
        countPullArms[arm]+=1
        totalRewardArms[arm] += tempReward
        empericalMeanArms[arm]=totalRewardArms[arm]/countPullArms[arm]
        totalEmpericalReward += tempReward
    regret = horizon * maxTrueMean - totalEmpericalReward
    return regret
