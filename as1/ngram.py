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

#unigram
print("Unigram model preplexity:")
modelU = unigram(train)
print("Train = ", modelU.perp(train))
print("Val = ", modelU.perp(val))
print("Test = ", modelU.perp(test), "\n")

#bigram
print("Bigram model preplexity:")
modelB = bigram(train)
print("Train = ", modelB.perp(train))
print("Val = ", modelB.perp(val))
print("Test = ", modelB.perp(test), "\n")

#trigram
print("Trigram model preplexity:")
modelT = trigram(train)
print("Train = ", modelT.perp(train))
print("val = ", modelT.perp(val))
print("Test = ", modelT.perp(test), "\n")

smoo = trigram(train, .1, .3, .6)
print("Perplexity for l1=.1, l2=.3, l3=.6")
print("Train = ", smoo.perp(train))
print("Val = ", smoo.perp(val))
print("Test = ", smoo.perp(test))
smoo = trigram(train, .2, .2, .6)
print("Perplexity for l1=.2, l2=.2 l3=.6")
print("Train = ", smoo.perp(train))
print("Val = ", smoo.perp(val))
print("Test = ", smoo.perp(test))
smoo = trigram(train, .1, .1, .8)
print("Perplexity for l1=.1, l2=.1, l3=.8")
print("Train = ", smoo.perp(train))
print("Val = ", smoo.perp(val))
print("Test = ", smoo.perp(test))
smoo = trigram(train, .8, .1, .1)
print("Perplexity for l1=.8, l2=.1, l3=.1")
print("Train = ", smoo.perp(train))
print("Val = ", smoo.perp(val))
print("Test = ", smoo.perp(test))
smoo = trigram(train, .1, .8, .1)
print("Perplexity for l1=.1, l2=.8, l3=.1")
print("Train = ", smoo.perp(train))
print("Val = ", smoo.perp(val))
print("Test = ", smoo.perp(test))


print("Trigram model preplexity (Half):")
modelT = trigram(train[:(len(train)/2 +1)])
print("Train = ", modelT.perp(train))
print("val = ", modelT.perp(val))
print("Test = ", modelT.perp(test), "\n")