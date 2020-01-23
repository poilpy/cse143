import numpy as np

start = "START"
stop = "STOP"

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

    # def perp(self, test):
    #     logProb = 0
    #     sentProb = 1
    #     for word in test:
    #         if word != stop:
    #             if word in self.freq:
    #                 if sentProb == 0:
    #                     print(word)
    #                     print(self.wordProb("induced"))
    #                 sentProb *= self.wordProb(word)
    #             else:
    #                 sentProb *= self.wordProb("UNK")
    #         else:
    #             print(sentProb)
    #             logProb -= math.log2(sentProb)
    #             sentProb = 1
    #     logProb = logProb/len(test)
    #     return math.pow(2, logProb)

    def perp(self, test):
        logProb = 0
        for word in test:
            if word in self.freq:
                logProb -= np.log2(self.wordProb(word))
            else:
                logProb -= np.log2(self.wordProb("UNK"))
        logProb = logProb/len(test)
        return np.power(2, logProb)

class bigram:
    def __init__(self, text):
        self.freq = dict()
        self.text = text
        self.total = len(text) - 2
        self.prevWord = None
        self.uni = unigram(text)
        for word in self.text:
            if self.prevWord != None:
                self.freq[(self.prevWord, word)] = self.freq.get((self.prevWord, word), 0) + 1
            self.prevWord = word
        self.numUnique = len(self.freq) - 2

    def wordProb(self, prevWord, word):
        num = self.freq.get((prevWord, word), 0)
        den = self.uni.freq[prevWord]
        return num/den
    
    def perp(self, test):
        logProb = 0
        self.prevWord = None
        for word in test:
            if word not in self.uni.freq:
                word = "UNK"
            if self.prevWord != None:
                logProb -= np.log2(self.wordProb(self.prevWord, word) if self.wordProb(self.prevWord, word) != 0 else 1)
            self.prevWord = word
        logProb = logProb/len(test)
        return np.power(2, logProb)

class trigram:
    def __init__(self, text):
        self.freq = dict()
        self.text = text
        self.total = len(text) - 2
        self.prevWord = None
        self.prevPrevWord = None
        self.bi = bigram(text)
        for word in self.text:
            if self.prevWord != None:
                if self.prevPrevWord != None:
                    self.freq[(self.prevPrevWord, self.prevWord, word)] = self.freq.get((self.prevPrevWord, self.prevWord, word), 0) + 1
                self.prevPrevWord = self.prevWord
            self.prevWord = word
        self.numUnique = len(self.freq) - 2

    def wordProb(self, prevPrevWord, prevWord, word):
        num = self.freq.get((prevPrevWord, prevWord, word), 0)
        den = self.bi.freq.get((prevPrevWord, prevWord), 0)
        if den == 0 or num == 0:
            # print(word)
            # print(num)
            # print("/")
            # print(den)
            return 1
        return np.float(num/den)
    
    def perp(self, test):
        logProb = 0
        self.prevWord = None
        self.prevPrevWord = None
        for word in test:
            if word not in self.bi.uni.freq:
                word = "UNK"
            if self.prevWord != None:
                if self.prevPrevWord != None:
                    logProb -= np.log2(self.wordProb(self.prevPrevWord, self.prevWord, word))
                self.prevPrevWord = self.prevWord
            self.prevWord = word
        logProb = logProb/len(test)
        return np.power(2, logProb)