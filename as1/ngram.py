from collections import Counter
from models import unigram

UNK = "UNK"

def openF(f):
    with open (f, "r", encoding="utf8") as myfile:
        data=myfile.read()
    arr = data.split()
    # return [x.lower() for x in arr]
    return arr

train = openF("1b_benchmark.train.tokens")
val = openF("1b_benchmark.dev.tokens")
test = openF("1b_benchmark.test.tokens")

print(train[0])
print(val[0])
print(test[0])

trainC = Counter(train)
train = [n if trainC[n] >= 3 else "UNK" for n in train]
print(train[0:100]) 
train.insert(0, "START")
train.append("STOP")
print(len(set(train)))

modelU = unigram(train)
modelU.printFreq(20)
