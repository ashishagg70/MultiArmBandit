import epsilonGreedy as eG
import ucb
import thompsonSampling as tS
import numpy as np
import time
import os
if __name__ == '__main__':
    instances=[[0.4, 0.8], [0.4, 0.3, 0.5, 0.2, 0.1], [0.15, 0.23, 0.37, 0.44, 0.5, 0.32, 0.78, 0.21, 0.82, 0.56, 0.34, 0.56, 0.84, 0.76, 0.43, 0.65, 0.73, 0.92, 0.1, 0.89, 0.48, 0.96, 0.6, 0.54, 0.49]
]
    epsilon=.02
    out1=['epsilon-greedy','ucb','kl-ucb', 'thompson-sampling']
    #out1KL = ['kl-ucb']
    out2=['thompson-sampling','thompson-sampling-with-hint']
    horizonList=[100, 400, 1600, 6400, 25600, 102400]
    #horizonListSmall = [100, 400]
    start_time = time.time()
    f1=open('outputDataT1.txt', 'w')
    #f1=open('/content/drive/My Drive/Colab Notebooks/MultiArmBandits/submission/outputDataT1.txt', 'w')
    for arms in instances:
        for algorithm in out1:
            for horizon in horizonList:
                for randomSeed in range(50):
                    regret = 0
                    if (algorithm == 'epsilon-greedy'):
                        regret = eG.epsilonGreedyAlgo(arms, randomSeed, epsilon, horizon)
                    elif (algorithm == 'ucb'):
                        regret = ucb.ucb(arms, randomSeed, horizon)
                    elif (algorithm == 'kl-ucb'):
                        regret = ucb.kl_ucb(arms, randomSeed, horizon)
                    elif (algorithm == 'thompson-sampling'):
                        regret = tS.thompsonSampling(arms, randomSeed, horizon)
                    out=""
                    if arms==instances[0]:
                        out="{}, {}, {}, {}, {}, {}{}".format("../instances/i-1.txt",algorithm, randomSeed, epsilon, horizon, regret,'\n')
                    elif arms==instances[1]:
                        out="{}, {}, {}, {}, {}, {}{}".format("../instances/i-2.txt",algorithm, randomSeed, epsilon, horizon, regret,'\n')
                    elif arms==instances[2]:
                        out="{}, {}, {}, {}, {}, {}{}".format("../instances/i-3.txt",algorithm, randomSeed, epsilon, horizon, regret,'\n')
                    f1.write(out)
    f1.close()
    f1 = open('outputDataT1.txt', 'r')
    lines=f1.read().rstrip('\n')
    f1.close()
    f1 = open('outputDataT1.txt', 'w')
    f1.write(lines)
    f1.close()
    print("--- %s seconds ---" % (time.time() - start_time))
    #open('/content/drive/My Drive/Colab Notebooks/MultiArmBandits/submission/outputDataT2.txt', 'w')
    f2 = open('outputDataT2.txt', 'w')
    for arms in instances:
        for algorithm in out2:
            for horizon in horizonList:
                for randomSeed in range(50):
                    regret = 0
                    if (algorithm == 'thompson-sampling'):
                        regret = tS.thompsonSampling(arms, randomSeed, horizon)
                        pass
                    elif(algorithm == 'thompson-sampling-with-hint'):
                        regret = tS.thompsonSamplingWithHint(arms, randomSeed, horizon, sorted(arms))
                    out = ""
                    if arms == instances[0]:
                        out = "{}, {}, {}, {}, {}, {}{}".format("../instances/i-1.txt", algorithm, randomSeed, epsilon,
                                                                horizon, regret, '\n')
                    elif arms == instances[1]:
                        out = "{}, {}, {}, {}, {}, {}{}".format("../instances/i-2.txt", algorithm, randomSeed, epsilon,
                                                                horizon, regret, '\n')
                    elif arms == instances[2]:
                        out = "{}, {}, {}, {}, {}, {}{}".format("../instances/i-3.txt", algorithm, randomSeed, epsilon,
                                                                horizon, regret, '\n')
                    f2.write(out)
    f2.close()
    f2 = open('outputDataT2.txt', 'r')
    lines = f2.read().rstrip('\n')
    f2.close()
    f2 = open('outputDataT2.txt', 'w')
    f2.write(lines)
    f2.close()
    print("--- %s seconds ---" % (time.time() - start_time))
