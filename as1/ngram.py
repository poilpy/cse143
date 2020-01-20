from collections import Counter
from models import unigram
from models import bigram

UNK = "UNK"

def openF(f):
    with open(f, "r", encoding="utf8") as myfile:
        data=myfile.read()
    arr = data.split()
    return arr

train = openF("1b_benchmark.train.tokens")
val = openF("1b_benchmark.dev.tokens")
test = openF("1b_benchmark.test.tokens")

trainC = Counter(train)
train = [n if trainC[n] >= 3 else "UNK" for n in train]
train.insert(0, "START")
train.append("STOP")

modelU = unigram(train)
print(modelU.perp(train))
print(modelU.perp(val))
print(modelU.perp(test))

modelB = bigram(train)
print(modelB.wordProb('IBM', 'spokesman'))
