from html2text import read_warc
from NER import ner,nlp
from EntityLinking import entity_linking,label_process
from RelationExtraction import load_setting,load_model,load_dataloader,predict_re
from nltk.tokenize import word_tokenize
from multiprocessing.pool import ThreadPool

# Press the green button in the gutter to run the script.
sentences = []
entity_links = []
#a dictionary for mapping our relation in wikidata
re_dict = {
        'Cause-Effect': "https://www.wikidata.org/wiki/Property:P828",
        'Instrument-Agency': "https://www.wikidata.org/wiki/Property:P2283",
        "Product-Producer": "https://www.wikidata.org/wiki/Property:P176",
        "Content-Container": "https://www.wikidata.org/wiki/Property:P527",
        "Entity-Origin": "https://www.wikidata.org/wiki/Property:P19",
        "Entity-Destination": "https://www.wikidata.org/wiki/Property:P1444",
        "Component-Whole": "https://www.wikidata.org/wiki/Property:P31",
        "Member-Collection": "https://www.wikidata.org/wiki/Property:P195",
        "Message-Topic": "https://www.wikidata.org/wiki/Property:P921"
    }

#aggrate the sentences in the same web page into one list
def aggragate(l):
    flag = l[0][0]
    sub = []
    new_l = []
    for item in l:
        if item[0] == flag:
            sub.append(item)
        else:
            flag = item[0]
            new_l.append(sub)
            sub = []
    new_l.append(sub)
    return new_l

#find the relation between any two entities in the same sentences
def find_relation(link):
    with open('predict_relation.txt','a') as f2:
        if len(link) > 1:
            key = link[0][0]
            print(key)
            # get all candidate entity set
            for i in range(len(link)):
                for j in range(len(link) - 1, i, -1):
                    if link[i][3] != link[j][3]:
                        continue
                    e1 = link[i][1][0]
                    e1_uri = link[i][2]
                    e2 = link[j][1][0]
                    e2_uri = link[j][2]
                    words = word_tokenize(str(link[i][3]))
                    if e1 not in words or e2 not in words:
                        # print(e1,e2,str(sent))
                        continue
                    data = str(e1) + '\t' + str(e2) + '\t' + str(link[i][3])
                    relation = predict_re(args, data_loader, model, data)
                    if relation[-6:-1] == 'e2,e1':
                        write_line = 'RELATION:' + '\t' + str(key) + '\t' + f"<{e2_uri}>" + '\t' + f"<{e1_uri}>" + '\t'\
                                     + relation[:-7] + '\t' +re_dict[relation[:-7]] + '\n'
                        f2.write(write_line)
                        print('RELATION:' + '\t' + str(key) + '\t' + f"<{e2_uri}>" + '\t' + f"<{e1_uri}>" + '\t' + relation[:-7]
                              + '\t' +re_dict[relation[:-7]])
                    elif relation[-6:-1] == 'e2,e1':
                        write_line = 'RELATION:' + '\t' + str(key) + '\t' + f"<{e1_uri}>" + '\t' + f"<{e2_uri}>" + '\t' \
                                     + relation[:-7] + '\t' + re_dict[relation[:-7]] + '\n'
                        f2.write(write_line)
                        print('RELATION:' + '\t' + str(key) + '\t' + f"<{e1_uri}>" + '\t' + f"<{e2_uri}>" + '\t' + relation[:-7]
                              + '\t' + re_dict[relation[:-7]])
                    else:
                        write_line = 'RELATION:' + '\t' + str(key) + '\t' + f"<{e1_uri}>" + '\t' + f"<{e2_uri}>" + '\t' \
                                     + 'other'+ '\n'
                        f2.write(write_line)
                        print('RELATION:' + '\t' + str(key) + '\t' + f"<{e1_uri}>" + '\t' + f"<{e2_uri}>" + '\t' + 'other')

if __name__ == '__main__':
    args, params = load_setting()
    data_loader, metric_labels = load_dataloader(args, params)
    model = load_model(data_loader, params)

    tup_list = [(key, text) for key, text in read_warc("data/sample.warc.gz")]
    #we take the first 200 pages to predict
    tup_list = tup_list[:100]
    for tup in tup_list:
        sents = nlp(tup[1]).sents
        for s in sents:
            # print((tup[0],s))
            sentences.append((tup[0], s))

    print('split sentence finished')

    file_path_1 = 'predict_entity_linking.txt'
    #file_path_2 = 'eva_entity_text.txt'
    with open(file_path_1,'a') as f1:
        for sent in sentences:
            for e in ner(str(sent[1])):
                link = entity_linking(e,sent[1])
                if link:
                    print((sent[0],e,link,sent[1]))
                    entity_links.append((sent[0],e, link,sent[1]))
                    f1.write('Entity:' + '\t' + str(sent[0]) + '\t' + str(e[0]) + '\t' + f"<{link}>"+ '\n')

    #for entity_link in entity_links:
    new_entity_links = aggragate(entity_links)
    #for entity_link in new_entity_links:
    pool_size = len(new_entity_links)
    pool = ThreadPool(pool_size)
    pool.map(find_relation, new_entity_links)
    pool.close()
    pool.join()
