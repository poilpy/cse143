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