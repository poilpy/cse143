

class unigram:
    def __init__(self, text):
        self.freq = dict()
        self.text = text
        self.total = len(text) - 2
        for word in self.text:
            self.freq[word] = self.freq.get(word, 0) + 1


    def printFreq(self, num):
        x = 0
        for word in self.freq:
            if x > num:
                break
            print(word)
            print(self.freq[word])
            x = x+1