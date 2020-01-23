from collections import Counter
from models import unigram
from models import bigram
from models import trigram

UNK = "UNK"

def openF(f):
    with open(f, "r", encoding="utf8") as myfile:
        data=myfile.read().replace("\n", " STOP Start")
    arr = data.split()
    arr.insert(0, "START")
    return arr[:-1]

train = openF("1b_benchmark.train.tokens")
val = openF("1b_benchmark.dev.tokens")
test = openF("1b_benchmark.test.tokens")

trainC = Counter(train)
train = [n if trainC[n] >= 3 else "UNK" for n in train]

print("Unigram model preplexity: Train/Val/Test")
modelU = unigram(train)
print(modelU.perp(train))
print(modelU.perp(val))
print(modelU.perp(test))

print("Bigram model preplexity: Train/Val/Test")
modelB = bigram(train)
print(modelB.perp(train))
print(modelB.perp(val))
print(modelB.perp(test))

print("Trigram model preplexity: Train/Val/Test")
modelT = trigram(train)
print(modelT.perp(train))
print(modelT.perp(val))
print(modelT.perp(test))