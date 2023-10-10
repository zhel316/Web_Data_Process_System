import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


#generating stop words list
def stop_word_union():
    stopWords = set(stopwords.words('english'))
    processedStopWords = set()
    for stopWord in stopWords:
        tokens = nltk.word_tokenize(stopWord)
        for token in tokens:
            processedStopWords.add(token)
    return processedStopWords

class WSD:
    def __init__(self, word, sentence):
        self.sentence = sentence
        self.word = word
        self.stopwords = stop_word_union()

    #split words without stopwords，number and other signs
    def LeskTokenize(self, s):
        result = []
        clean_string = ''
        for c in s:
            if c.isalpha() or c == ' ':
                clean_string =clean_string + c
        raw_tokens = nltk.word_tokenize(clean_string)
        for raw_token in raw_tokens:
            if raw_token in raw_tokens:
                result.append(raw_token)
        return result

    #caculating how many words are same between two words list
    def overlap(self, a, b):
        result = 0
        if len(a) == 0 or len(b) == 0:
            return result
        for x in a:
            for y in b:
                if x == y:
                    result += 1
        return result


    #Lesk algorithm
    #@input: one word need disambiguisation, one sentence contain this word
    #@output: the meaning of the word in this sentence
    def LeskDisambiguisation(self):
        tokens = self.LeskTokenize(self.sentence)

        if len(tokens) == 0:
            #raise Exception("Sorry, there si nothing in the sentence")
            return ''
        if self.word not in tokens:
            #raise Exception('Sorry, the word is not in sentence')
            return ''
        context = set()
        for token in tokens:
            if token != self.word:
                context.add(token)
        if len(context) == 0:
            #raise Exception("Sorry, the word that needs desambiguation is the only word in the sentence.")
            return ''

        wn.ensure_loaded()  # first access to wn transforms it

        if not wn.synsets(self.word):
            #print('Sorry, we can not find this word')
            return ''

        contextDefinitions = []
        for v in context:
            for i in range(len(wn.synsets(v))):
                definitionWords = self.LeskTokenize(wn.synset(wn.synsets(v)[i].name()).definition())
                for definitionWord in definitionWords:
                    contextDefinitions.append(definitionWord)

        maxValue = 0
        maxNumber = 0

        for j in range(len(wn.synsets(self.word))):
            definitionWords = self.LeskTokenize(wn.synset(wn.synsets(self.word)[j].name()).definition())
            currentValue = self.overlap(definitionWords, contextDefinitions)
            if currentValue > maxValue:
                maxValue = currentValue
                maxNumber = j


        return wn.synset(wn.synsets(self.word)[maxNumber].name()).definition()


