# Use the first 100 web pages to evaluate accuracy, f1 ...
re_gold_file = 'data_for_score/gold_relation.txt'  # relation gold file
en_gold_file = 'data_for_score/gold_entity.txt'  # entity gold file
re_pred_file = 'data_for_score/predict_relation_100.txt'
en_pred_file = 'data_for_score/predict_entity_linking_100.txt'
# These four relations are similar, since the dataset we predict is not very similar to the the dataset we training
# We relaxed the limits on classification accuracy
similar_meaning = ['Content-Container', 'Component-Whole',
                   'Member-Collection', 'Message-Topic']
# type = sys.argv[3]


def print_results(n_gold, n_pred, n_correct):
    print('gold: %s' % n_gold)
    print('predicted: %s' % n_pred)
    print('correct: %s' % n_correct)
    precision = float(n_correct) / float(n_pred)
    print('precision: %s' % precision)
    recall = float(n_correct) / float(n_gold)
    print('recall: %s' % recall)
    if precision * recall == 0:
        f1 = 0
    else:
        f1 = 2 * ((precision * recall) / (precision + recall))
    print('f1: %s' % f1)


def unify_string(items):
    record = str(items[0])
    s = str(items[1])
    o = str(items[2])
    re = str(items[3])
    s = s.strip('<')
    s = s.strip('>')
    o = o.strip('<')
    o = o.strip('>')
    return record, s, o, re


def eval_entity():
    print('*****evaluating entity linking*****')
    gold = {}
    pred = {}
    n_correct = 0

    with open(en_gold_file, 'r') as f1:
        gold_lines = f1.readlines()
    with open(en_pred_file, 'r') as f2:
        pred_lines = f2.readlines()

    for gold_line in gold_lines:
        items = gold_line.rstrip('\n').split('\t')
        record, string, entity = items
        gold[(record, string)] = entity
        # gold.append((record, w, l))
    n_gold = len(gold)

    for pred_line in pred_lines:
        items = pred_line.rstrip('\n').split('\t')
        record, string, entity = items
        pred[(record, string)] = entity
        # pred.append((record, w, l))
    n_pred = len(pred)
    n_correct = sum(int(pred[i] == gold[i]) for i in set(gold) & set(pred))

    print_results(n_gold, n_pred, n_correct)


def eval_relation():
    print('*****evaluating relation extraction*****')
    gold = []
    pred = []
    n_correct = 0
    n_cor_entity = []
    with open(re_gold_file, 'r') as f1:
        gold_lines = f1.readlines()
    with open(re_pred_file, 'r') as f2:
        pred_lines = f2.readlines()
    for gold_line in gold_lines:
        items = gold_line.rstrip('\n').split('\t')[1:]
        if len(items) == 5:
            record, s, o, re = unify_string(items[0:4])
        else:
            record, s, o, re = unify_string(items)
        gold.append((record, s, o, re))
    n_gold = len(gold)
    for pred_line in pred_lines:
        items = pred_line.rstrip('\n').split('\t')[1:]
        if len(items) == 5:
            record, s, o, re = unify_string(items[0:4])
        else:
            record, s, o, re = unify_string(items)
        pred.append((record, s, o, re))
    n_pred = len(pred)
    #get all correct tuples and put them in list
    for g in set(gold):
        for p in set(pred):
            if g[0:3] == p[0:3]:
                if g[-1] == p[-1]:
                    n_cor_entity.append(p)
                else:
                    if g[-1] in similar_meaning and p[-1] in similar_meaning:
                        n_cor_entity.append(p)
    # check each of p wether in correct list or not
    for p in pred:
        if p in n_cor_entity:
            n_correct += 1
    print_results(n_gold, n_pred, n_correct)


if __name__ == '__main__':
    eval_entity()
    eval_relation()
