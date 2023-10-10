#This file is used to generate gold file for relation by r-bert model
from predict import predict_re as pre
from nltk.tokenize import word_tokenize

# Press the green button in the gutter to run the script.

#This function is to collect sentences which are in same web page
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

if __name__ == '__main__':
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
    items = []
    with open('data_for_score/eva_entity_text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            items.append(line.split('\t')[1:])
    new_items = aggragate(items)
    with open('data_for_score/gold_relation.txt', 'w') as f:
        for item in new_items:
            if len(item)>0:
                key = item[0][0]
                print(key)
                for i in range(len(item)):
                    for j in range(len(item)-1, i,-1):
                        if item[i][3] == item[j][3] and item[i][1]!=item[j][1]:
                            sent = item[i][3]
                            sent = sent.rstrip('\n')
                            e1 = item[i][1]
                            e2 = item[j][1]
                            s_list = word_tokenize(str(sent))
                            if e1 not in s_list or e2 not in s_list:
                                # print(e1,e2,str(sent))
                                continue
                            e1_s = s_list.index(str(e1))  # zhe's
                            e1_e = e1_s + 2  # zhe's
                            s_list.insert(e1_s, '<e1>')  # zhe's
                            s_list.insert(e1_e, '</e1>')  # zhe's
                            e2_s = s_list.index(str(e2))  # zhe's
                            e2_e = e2_s + 2  # zhe's
                            s_list.insert(e2_s, '<e2>')  # zhe's
                            s_list.insert(e2_e, '</e2>')  # zhe's
                            #turn to the specified format that r-bert model need
                            data = ' '.join(str(v) for v in s_list)
                            relation = pre(data)
                            print(relation)
                            if relation[-8:-1] == '(e2,e1)':
                                write_line = 'RELATION:' + '\t' + key + '\t' + f"{item[j][2]}" + '\t' + f"<{item[i][2]}>" + '\t' \
                                             + relation[:-8] + '\t' + re_dict[relation[:-8]]
                                f.write(write_line + '\n')
                            elif relation[-8:-1] == '(e1,e2)':
                                write_line = 'RELATION:' + '\t' + key + '\t' + f"{item[i][2]}" + '\t' + f"<{item[j][2]}>" + '\t' \
                                             + relation[:-8] + '\t' + re_dict[relation[:-8]]
                                f.write(write_line + '\n')
                            else:
                                write_line = 'RELATION:' + '\t' + key + '\t' + f"{item[i][2]}" + '\t' + f"<{item[j][2]}>" + '\t' \
                                             + 'other'
                                f.write(write_line + '\n')



