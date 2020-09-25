from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

ts1=defaultdict(list)
tsh1=defaultdict(list)
ts2=defaultdict(list)
tsh2=defaultdict(list)
ts3=defaultdict(list)
tsh3=defaultdict(list)

for line in open('outputDataT2.txt', 'r'):
    token = [l.strip() for l in line.split(',')]
    if(token[0]=='../instances/i-1.txt'):
        if(token[1]=='thompson-sampling-with-hint'):
            tsh1[int(token[4])].append(float(token[5]))
        elif (token[1] == 'thompson-sampling'):
            ts1[int(token[4])].append(float(token[5]))

    elif(token[0]=='../instances/i-2.txt'):
        if (token[1] == 'thompson-sampling-with-hint'):
            tsh2[int(token[4])].append(float(token[5]))
        elif (token[1] == 'thompson-sampling'):
            ts2[int(token[4])].append(float(token[5]))
    elif (token[0] == '../instances/i-3.txt'):
        if (token[1] == 'thompson-sampling-with-hint'):
            tsh3[int(token[4])].append(float(token[5]))
        elif (token[1] == 'thompson-sampling'):
            ts3[int(token[4])].append(float(token[5]))

keys=[100, 400, 1600, 6400, 25600, 102400]
logKeys=np.log(keys)

plt.xlabel('horizon(log scale)')
plt.ylabel('mean regret over 0-49 seeds')
plt.title('instance1_T2')
plt.semilogx(keys, [ np.array(tsh1[x]).mean() for x in keys], label='thompson-sampling-with-hint',color='lightblue', linewidth=2)
plt.semilogx(keys, [ np.array(ts1[x]).mean() for x in keys], label='thompson-sampling',color='green', linewidth=2)
plt.legend(loc='upper left')
plt.show()
plt.clf()

plt.xlabel('horizon(log scale)')
plt.ylabel('mean regret over 0-49 seeds')
plt.title('instance2_T2')
plt.semilogx(keys, [ np.array(tsh2[x]).mean() for x in keys], label='thompson-sampling-with-hint',color='lightblue', linewidth=2)
plt.semilogx(keys, [ np.array(ts2[x]).mean() for x in keys], label='thompson-sampling',color='green', linewidth=2)
plt.legend(loc='upper left')
plt.show()
plt.clf()

plt.xlabel('horizon(log scale)')
plt.ylabel('mean regret over 0-49 seeds')
plt.title('instance3_T2')
plt.semilogx(keys, [ np.array(tsh3[x]).mean() for x in keys], label='thompson-sampling-with-hint',color='lightblue', linewidth=2)
plt.semilogx(keys, [ np.array(ts3[x]).mean() for x in keys], label='thompson-sampling',color='green', linewidth=2)
plt.legend(loc='upper left')
plt.show()
plt.clf()
