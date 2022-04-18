import argparse
import json
from collections import defaultdict


preposition_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                    'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                    'inside', 'outside', 'into', 'around']

preposition_dict = {k: v for v, k in enumerate(preposition_list)}


def load_data(in_f):
    with open(in_f, 'r') as f:
        data = f.readlines()

    data = [json.loads(x.strip()) for x in data]
    return data


def dump_data(docs, out_f):
    with open(out_f, 'w') as f:
        for doc in docs:
            json.dump(doc, f)
            f.write('\n')


def data_conversion(doc):
    tokenized_entities = doc['nps']
    links = doc['np_relations']

    entity_spans = {}
    for entity_id, row in tokenized_entities.items():
        if len(row) == 1: continue
        entity_spans[entity_id] = (row['first_token'], row['last_token'])

    links_dic = defaultdict(list)
    prepositions_dic = defaultdict(dict)
    extended_prep_dic = defaultdict(dict)
    for row in links:
        links_dic[row['anchor']].append(row['complement'])
        prepositions_dic[row['anchor']][row['complement']] = preposition_dict[row['preposition']]

        if row['anchor'] not in extended_prep_dic or row['complement'] not in extended_prep_dic[row['anchor']]:
            extended_prep_dic[row['anchor']][row['complement']] = [preposition_dict[row['preposition']]]
        else:
            extended_prep_dic[row['anchor']][row['complement']].append(preposition_dict[row['preposition']])

    # Building a vector in the size of N^2, for all possible links
    # This includes the links on the diagonal (between the same NP), which will be filtered
    # in the model
    link_labels = []
    extended_prep_labels = []
    for ind1, span_id1 in enumerate(entity_spans.keys()):
        for ind2, span_id2 in enumerate(entity_spans.keys()):
            if span_id1 in links_dic and span_id2 in links_dic[span_id1]:
                link_labels.append(1)
                extended_prep_labels.append(extended_prep_dic[span_id1][span_id2])
            else:
                extended_prep_labels.append([preposition_dict['no-relation']])
                link_labels.append(0)
            if ind1 == ind2:
                link_labels[-1] = -1

    dic = {
        'id': doc['id'],
        'prepositions': extended_prep_labels,
        'relations': link_labels,
    }

    return dic


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--input", type=str, help="input data file", default="train")
    parse.add_argument("--output", type=str, help="output file where the converted values will be written",
                       default="train")

    args = parse.parse_args()

    print('=== Reading input file ===')

    document_answers = load_data(args.input)

    print('=== Converting... ===')
    minimal_labels = [data_conversion(x) for x in document_answers]

    print('=== Writing results to file ===')
    dump_data(minimal_labels, args.output)

    print('=== Done ===')


if __name__ == '__main__':
    main()
