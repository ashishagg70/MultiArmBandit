import customParser as parser
import epsilonGreedy as eG
import ucb
import numpy as np
if __name__ == '__main__':
    arms = parser.getArg('instance')  # float array
    algorithm = parser.getArg('algorithm')
    randomSeed = int(parser.getArg('randomSeed'))
    epsilon = float(parser.getArg('epsilon'))
    horizon = int(parser.getArg('horizon'))
    regret=np.zeros(50)
    if(algorithm=='epsilon-greedy'):
        for seed in range(50):
            regret[seed]=eG.epsilonGreedyAlgo(arms, seed, epsilon, horizon)
        parser.printOutput(regret.mean())
    elif(algorithm=='ucb'):
        for seed in range(50):
            regret[seed]=ucb.ucb(arms, seed, horizon)
        parser.printOutput(regret.mean())


