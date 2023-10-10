import en_core_web_md
from html2text import read_warc
nlp = en_core_web_md.load()

NER_type = ["DATE","TIME","CARDINAL","ORDINAL","QUANTITY","PERCENT","MONEY"]


#using spacy library to process the NER task, automatically lable the entities in text
def ner(text):
    doc = nlp(text)
    entity = [(X.text, X.label_) for X in doc.ents if X.label_ not in NER_type]
    entity = list(set(entity))
    return entity

if __name__ == '__main__':
    for k, text in read_warc("data/sample.warc.gz"):
        print(ner(text))