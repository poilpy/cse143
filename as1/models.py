import math

start = "START"
stop = "STOP"

#loop through the text and get the word frequencies
class unigram:
    def __init__(self, text):
        self.freq = dict()
        self.text = text
        self.total = len(text) - 2
        for word in self.text:
            self.freq[word] = self.freq.get(word, 0) + 1
        self.numUnique = len(self.freq) - 2
#calculate word probability from frequency
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
#calculate perplexity using formula
    def perp(self, test):
        logProb = 0
        for word in test:
            if word in self.freq:
                logProb -= math.log2(self.wordProb(word))
            else:
                logProb -= math.log2(self.wordProb("UNK"))
        logProb = logProb/len(test)
        return math.pow(2, logProb)
#loop through the text and get the word frequencies for bigram
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
#calculate word probability from frequency
    def wordProb(self, prevWord, word):
        # x = 10
        # for a in self.freq:
        #     # if x > 10:
        #     #     break
        #     print(a)
        #     x = x+1
        num = self.freq.get((prevWord, word), 0)
        den = self.uni.freq[prevWord]
        return num/den
  #calculate perplexity using formula 
    def perp(self, test):
        logProb = 0
        self.prevWord = None
        for word in test:
            if word not in self.uni.freq:
                word = "UNK"
            if self.prevWord != None:
                logProb -= math.log2(self.wordProb(self.prevWord, word) if self.wordProb(self.prevWord, word) else 1)
            self.prevWord = word
        logProb = logProb/len(test)
        return math.pow(2, logProb)
