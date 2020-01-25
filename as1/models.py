import numpy as np

start = "START"
stop = "STOP"

#loop through the text and get the word frequencies
class unigram:
    def __init__(self, text):
        self.freq = dict()
        self.total = 0
        for word in text:
            self.freq[word] = self.freq.get(word, 0) + 1
            if word != stop or word != start: 
                self.total += 1

#calculate word probability from frequency
    def wordProb(self, word):
        num = self.freq.get(word, 0)
        den = self.total
        return num/den

#calculate perplexity using formula
    def perp(self, test):
        logProb = 0
        for word in test:
            if word not in self.freq:
                word = "UNK"
            if word != start:
                logProb -= np.log2(self.wordProb(word))
        logProb = logProb/len(test)
        return np.power(2, logProb)
        
#loop through the text and get the word frequencies for bigram
class bigram:
    def __init__(self, text):
        self.freq = dict()
        self.prevWord = None
        self.uni = unigram(text)
        for word in text:
            if self.prevWord != None:
                self.freq[(self.prevWord, word)] = self.freq.get(
                    (self.prevWord, word), 0) + 1
            self.prevWord = word

#calculate word probability from frequency
    def wordProb(self, prevWord, word):
        num = self.freq.get((prevWord, word), 0)
        den = self.uni.freq.get(prevWord, 0)
        if den == 0:
            return 0
        return num/den

  #calculate perplexity using formula 
    def perp(self, test):
        logProb = 0
        self.prevWord = None
        for word in test:
            if word not in self.uni.freq:
                word = "UNK"
            if self.prevWord != None and word != start:
                prob = self.wordProb(self.prevWord, word)
                if prob == 0:
                    return "INF"
                logProb -= np.log2(prob)
            self.prevWord = word
        logProb = logProb/len(test)
        return np.power(2, logProb)
    
#loop through the text and get the word frequencies
class trigram:
    def __init__(self, text, l1=1, l2=1, l3=1):
        self.freq = dict()
        self.prevWord = None
        self.prevPrevWord = None
        self.bi = bigram(text)
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        for word in text:
            if self.prevWord != None and word != start:
                if self.prevPrevWord != None:
                    self.freq[(self.prevPrevWord, self.prevWord, word)] = self.freq.get(
                        (self.prevPrevWord, self.prevWord, word), 0) + 1
                self.prevPrevWord = self.prevWord
            self.prevWord = word
            
#calculate word probability from frequency
    def wordProb(self, prevPrevWord, prevWord, word):
        num = self.freq.get((prevPrevWord, prevWord, word), 0)
        den = self.bi.freq.get((prevPrevWord, prevWord), 0)
        if den == 0:
            return 0
        return np.float(num/den)
    #calculate perplexity using formula 
    def perp(self, test):
        logProb = 0
        self.prevWord = None
        self.prevPrevWord = None
        for word in test:
            if word not in self.bi.uni.freq:
                word = "UNK"
            if self.prevWord != None:
                if self.prevPrevWord != None:
                    if self.l3 != 1:
                        logProb -= np.log2(self.l3 * self.wordProb(self.prevPrevWord, self.prevWord, word)
                            + self.l2 * self.bi.wordProb(self.prevWord, word)
                            + self.l1 * self.bi.uni.wordProb(word)
                            )
                    else:
                        prob = self.wordProb(self.prevPrevWord, self.prevWord, word)
                        if prob == 0:
                            return "INF"
                        logProb -= np.log2(prob)
                self.prevPrevWord = self.prevWord
            self.prevWord = word
        logProb = logProb/len(test)
        return np.power(2, logProb)
