import numpy as np
from gensim.models import Word2Vec

class EntityRank:
    def __init__(self,ent,results):
        self.ent = ent
        self.results = results
        #this word2vec model we pre-trained on google colab
        #embedding words into vector
        #this word2vec model is skip-gram since the sentences are not very long
        #and consider the content
        self.model = Word2Vec.load('w2v_model/text8_w2v.model')


    #caculating the cosine similarity and find the biggest one
    def cal(self):
        max_flag = 0
        max_value = 0
        for index,result in enumerate(self.results):
            if result:
                simi_value = self.compute_cosine_similarity(self.ent, result)
                if simi_value > max_value:
                    max_flag = index
                    max_value = simi_value
        return max_flag


    def build_sentence_vector(self,sentence, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0
        for word in sentence:
            try:
                vec += self.model.wv[word].reshape((1, size))
                count += 1
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec


    def cosine_similarity(self,vec1, vec2):
        a = np.array(vec1, dtype='float64')
        b = np.array(vec2, dtype='float64')
        cos1 = np.sum(a * b)
        cos21 = np.sqrt(sum(a ** 2))
        cos22 = np.sqrt(sum(b ** 2))
        cosine_value = cos1 / float(np.dot(cos21,cos22))
        return cosine_value

    def compute_cosine_similarity(self,sents_1, sents_2):
        size = 100
        vec1 = self.build_sentence_vector(sents_1, size)
        vec2 = self.build_sentence_vector(sents_2, size)
        similarity = self.cosine_similarity(vec1, vec2)
        return similarity

