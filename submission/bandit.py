import customParser as parser
import epsilonGreedy as eG
import ucb
import thompsonSampling as tS
if __name__ == '__main__':
    arms = parser.getArg('instance')  # float array
    algorithm = parser.getArg('algorithm')
    randomSeed = int(parser.getArg('randomSeed'))
    epsilon = float(parser.getArg('epsilon'))
    horizon = int(parser.getArg('horizon'))
    regret=0
    if(algorithm=='epsilon-greedy'):
        regret=eG.epsilonGreedyAlgo(arms, randomSeed, epsilon, horizon)
    elif(algorithm=='ucb'):
        regret=ucb.ucb(arms, randomSeed, horizon)
    elif (algorithm == 'kl-ucb'):
        regret = ucb.kl_ucb(arms, randomSeed, horizon)
    elif (algorithm == 'thompson-sampling'):
        regret = tS.thompsonSampling(arms, randomSeed, horizon)
    elif (algorithm == 'thompson-sampling-with-hint'):
        regret = tS.thompsonSamplingWithHint(arms, randomSeed, horizon, sorted(arms))
    parser.printOutput(regret)
