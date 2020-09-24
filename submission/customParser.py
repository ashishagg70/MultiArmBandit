import argparse


#filepath=sys.argv[1]

ap = argparse.ArgumentParser()

ap.add_argument("--instance", required=True,
   help="first operand", default='../instances/i-1.txt')

ap.add_argument("--algorithm", required=True,
   help="second operand", default='epsilon-greedy')

ap.add_argument("--randomSeed", required=True,
   help="third operand", default='49')

ap.add_argument("--epsilon", required=True,
   help="fourth operand", default='0.02')

ap.add_argument("--horizon", required=True,
   help="fifth operand", default='1600')
args = vars(ap.parse_args())

def getArg(name):
   if(name=='instance'):
      filename=args[name]
      with open(filename) as f:
         content = f.readlines()
      content = [float(x.strip()) for x in content]
      return content

   return args[name]

def printOutput(regret):
   print("%s, %s, %s, %s, %s, %s"%(args['instance'], args['algorithm'], args['randomSeed'], args['epsilon'], args['horizon'],regret))


