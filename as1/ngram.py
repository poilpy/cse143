from collections import Counter
from models import unigram
from models import bigram
from models import trigram

UNK = "UNK"
#open file and return array of content
def openF(f):
    with open(f, "r", encoding="utf8") as myfile:
        data=myfile.read().replace("\n", " STOP Start ")
    arr = data.split()
    arr.insert(0, "START")
    return arr[:-1]
    
#data sets used for processing
train = openF("1b_benchmark.train.tokens")
val = openF("1b_benchmark.dev.tokens")
test = openF("1b_benchmark.test.tokens")

#replace tokens that happened less that three times with "UNKS"
trainC = Counter(train)
train = [n if trainC[n] >= 3 else UNK for n in train]

def printModel(model):
    print("Train = ", model.perp(train))
    print("Val = ", model.perp(val))
    print("Test = ", model.perp(test), "\n")


#unigram
print("Unigram model preplexity:")
modelU = unigram(train)
printModel(modelU)

#bigram
print("Bigram model preplexity:")
modelB = bigram(train)
printModel(modelB)

#trigram
print("Trigram model preplexity:")
modelT = trigram(train)
printModel(modelT)

smoo = trigram(train, .1, .3, .6)
print("Perplexity for l1=.1, l2=.3, l3=.6")
printModel(smoo)
smoo = trigram(train, .2, .2, .6)
print("Perplexity for l1=.2, l2=.2 l3=.6")
printModel(smoo)
smoo = trigram(train, .1, .1, .8)
print("Perplexity for l1=.1, l2=.1, l3=.8")
printModel(smoo)
smoo = trigram(train, .8, .1, .1)
print("Perplexity for l1=.8, l2=.1, l3=.1")
printModel(smoo)
smoo = trigram(train, .1, .8, .1)
print("Perplexity for l1=.1, l2=.8, l3=.1")
printModel(smoo)

print("Trigram model preplexity (Half):")
modelT = trigram(train[:int(len(train)/2)])
printModel(modelT)