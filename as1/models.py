import math

class unigram:
    def __init__(self, text):
        self.freq = dict()
        self.text = text
        self.total = len(text) - 2
        for word in self.text:
            self.freq[word] = self.freq.get(word, 0) + 1
        self.numUnique = len(self.freq) - 2

    def wordProb(self, word):
        num = self.freq[word]
        den = self.total
        return num/den

    def perp(self, test):
        logProb = 0
        for word in test:
            if word in self.freq:
                logProb -= math.log2(self.wordProb(word))
            else:
                logProb -= math.log2(self.wordProb("UNK"))
        logProb = logProb/len(test)
        return math.pow(2, logProb)

class bigram:
    def __init__(self, text):
        self.freq = dict()
        self.grams = set()
        self.text = text
        self.total = len(text) - 2
        self.prevWord = None
        self.uni = unigram(text)
        for word in self.text:
            if self.prevWord != None:
                self.freq[(self.prevWord, word)] = self.freq.get((self.prevWord, word), 0) + 1
                self.grams.add((self.prevWord, word))
            self.prevWord = word
        self.numUnique = len(self.freq) - 2

    def wordProb(self, prevWord, word):
        # x = 10
        # for a in self.freq:
        #     # if x > 10:
        #     #     break
        #     print(a)
        #     x = x+1
        num = self.freq[(prevWord, word)]
        den = self.uni.freq[prevWord]
        return num/den
    
    def perp(self, test):
        logProb = 0
        prevWord = None
        for word in test:
            if word in self.freq:
                logProb -= math.log2(self.wordProb(word))
            else:
                logProb -= math.log2(self.wordProb("UNK"))
        logProb = logProb/len(test)
        return math.pow(2, logProb)