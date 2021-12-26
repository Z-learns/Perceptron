import random
import math


# Initialize empty lists for training data and weight
trainingData = []
nIters = 50000
SSE = []
w0 = random.uniform(-2,2)
w1 = random.uniform(-2,2)
w2 = random.uniform(-2,2)
eta = 0.005
testData = []
SSE = []
yBar = []
yExample = []
acc = 0
print(w0, w1, w2)


"""
Alternate way of opening a file 

with open("training.txt") as f:
	listing = f.readlines()
	listing = [item.rstrip().split("\t") for item in listing]
	listing = [(1,int(item[0]) , int(item[1]), int(item[2])) for item in listing]
	#print(listing)
	#listing = [ tuple((1, tuple(map(int, items)) ))for items in listing]
	#print(listing)
"""

with open("train.txt") as f:
	listing = f.readlines()
	listing = [item.rstrip().split("\t") for item in listing]
	[item.insert(0,1) for item in listing]
	listing = [ tuple(map(int, items)) for items in listing ]
#print(listing)

for _ in range (nIters):
	for w,x,y,z in listing:
		logit = (w0*w + w1*x + w2*y)
		yCap = 1/(1+math.exp(-logit))
		#print(yCap)
		error = (z - yCap)**2
		#SSE.append(error)
		delta = (z - yCap) * yCap * (1-yCap) # ignored the constant -2
		w0 = w0 + (eta * w * delta)
		w1 = w1 + (eta * x * delta)
		w2 = w2 + (eta * y * delta)

print(w0, w1, w2)

# Validate model
with open("valid.txt") as f:
	Data = f.readlines()
	Data = [item.rstrip().split("\t") for item in Data]
	[item.insert(0,1) for item in Data]
	Data = [ tuple(map(int, items)) for items in Data]

for w,x,y,z in Data:
	yExample.append(z)
	pred = (w0*w + w1*x + w2*y)
	predict = 1/(1+math.exp(-pred))
    #print(predict)
	err = (z - predict)**2
	SSE.append(err)

	if (predict >= 0.5):
		yBar.append(1)
	else:
		yBar.append(0)


print(sum(SSE))
print(yBar)
for i in range(len(Data)):
    if (yBar[i]==yExample[i]):
        acc += 1
accuracy = (acc/len(Data))*100
print(accuracy)
