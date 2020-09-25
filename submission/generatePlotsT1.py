from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

ep1=defaultdict(list)
ucb1=defaultdict(list)
klUcb1=defaultdict(list)
ts1=defaultdict(list)
ep2=defaultdict(list)
ucb2=defaultdict(list)
klUcb2=defaultdict(list)
ts2=defaultdict(list)
ep3=defaultdict(list)
ucb3=defaultdict(list)
klUcb3=defaultdict(list)
ts3=defaultdict(list)

for line in open('outputDataT1.txt', 'r'):
    token = [l.strip() for l in line.split(',')]
    if(token[0]=='../instances/i-1.txt'):
        if(token[1]=='epsilon-greedy'):
            ep1[int(token[4])].append(float(token[5]))
        elif(token[1]=='ucb'):
            ucb1[int(token[4])].append(float(token[5]))
        elif (token[1] == 'kl-ucb'):
            klUcb1[int(token[4])].append(float(token[5]))
        elif (token[1] == 'thompson-sampling'):
            ts1[int(token[4])].append(float(token[5]))

    elif(token[0]=='../instances/i-2.txt'):
        if (token[1] == 'epsilon-greedy'):
            ep2[int(token[4])].append(float(token[5]))
        elif (token[1] == 'ucb'):
            ucb2[int(token[4])].append(float(token[5]))
        elif (token[1] == 'kl-ucb'):
            klUcb2[int(token[4])].append(float(token[5]))
        elif (token[1] == 'thompson-sampling'):
            ts2[int(token[4])].append(float(token[5]))
    elif (token[0] == '../instances/i-3.txt'):
        if (token[1] == 'epsilon-greedy'):
            ep3[int(token[4])].append(float(token[5]))
        elif (token[1] == 'ucb'):
            ucb3[int(token[4])].append(float(token[5]))
        elif (token[1] == 'kl-ucb'):
            klUcb3[int(token[4])].append(float(token[5]))
        elif (token[1] == 'thompson-sampling'):
            ts3[int(token[4])].append(float(token[5]))

keys=[100, 400, 1600, 6400, 25600, 102400]
logKeys=np.log(keys)

plt.xlabel('horizon (log scale)')
plt.ylabel('mean regret over 0-49 seeds')
plt.title('instance1_T1')
plt.semilogx(keys, [ np.array(ep1[x]).mean() for x in keys], label='eplison-greedy',color='lightblue', linewidth=2)
plt.semilogx(keys, [ np.array(ucb1[x]).mean() for x in keys],label='ucb', color='red', linewidth=2)
plt.semilogx(keys, [ np.array(klUcb1[x]).mean() for x in keys], label='kl-ucb',color='orange', linewidth=2)
plt.semilogx(keys, [ np.array(ts1[x]).mean() for x in keys], label='thompson-sampling',color='green', linewidth=2)
'''plt.plot(logKeys, [ np.array(ep1[x]).mean() for x in keys], label='eplison-greedy',color='lightblue', linewidth=2)
plt.plot(logKeys, [ np.array(ucb1[x]).mean() for x in keys],label='ucb', color='red', linewidth=2)
plt.plot(logKeys, [ np.array(klUcb1[x]).mean() for x in keys], label='kl-ucb',color='orange', linewidth=2)
plt.plot(logKeys, [ np.array(ts1[x]).mean() for x in keys], label='thompson-sampling',color='green', linewidth=2)'''
plt.legend(loc='upper left')
plt.show()
plt.clf()

plt.xlabel('horizon (log scale)')
plt.ylabel('mean regret over 0-49 seeds')
plt.title('instance2_T1')
plt.semilogx(keys, [ np.array(ep2[x]).mean() for x in keys], label='eplison-greedy',color='lightblue', linewidth=2)
plt.semilogx(keys, [ np.array(ucb2[x]).mean() for x in keys],label='ucb', color='red', linewidth=2)
plt.semilogx(keys, [ np.array(klUcb2[x]).mean() for x in keys], label='kl-ucb',color='orange', linewidth=2)
plt.semilogx(keys, [ np.array(ts2[x]).mean() for x in keys], label='thompson-sampling',color='green', linewidth=2)
plt.legend(loc='upper left')
plt.show()
plt.clf()

plt.xlabel('horizon (log scale)')
plt.ylabel('mean regret over 0-49 seeds')
plt.title('instance3_T1')
plt.semilogx(keys, [ np.array(ep3[x]).mean() for x in keys], label='eplison-greedy',color='lightblue', linewidth=2)
plt.semilogx(keys, [ np.array(ucb3[x]).mean() for x in keys],label='ucb', color='red', linewidth=2)
plt.semilogx(keys, [ np.array(klUcb3[x]).mean() for x in keys], label='kl-ucb',color='orange', linewidth=2)
plt.semilogx(keys, [ np.array(ts3[x]).mean() for x in keys], label='thompson-sampling',color='green', linewidth=2)
plt.legend(loc='upper left')
plt.show()
plt.clf()
