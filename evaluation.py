import os
os.chdir('seq2rel')
from seq2rel import Seq2Rel
from seq2rel.common import util
from allennlp.common.file_utils import cached_path
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='model.tar.gz',
                    help="a fine-tuned model file name")

parser.add_argument('--test_file', type=str, default='n-ary/test.txt',
                    help="a path to preprocessed test file")

parser.add_argument('--ner', type=bool, default=False,
                    help="evaluation on NER results")

parser.add_argument('--metric', type=str, default=None,
                    help="evaluation metric, [positive_combination_f1] or [any_combination_f1]")

args = parser.parse_args()

# load trained seq2rel model
model = args.model
seq2rel = Seq2Rel(model)

# do NER-RE evaluation
if args.ner:
    predictions=[]
    gold=[]
    
    # make predictions
    with open(cached_path(args.test_file), "r") as data_file:
        for line_num, line in enumerate(data_file):
            line = line.strip("\n")
            line_parts = line.split('\t')
            input_text = line_parts[0]
            gold.append(line_parts[1])
            gold_relations = [line_parts[1]]
            predicted_relations = seq2rel(input_text)
            predicted_relations = [i.replace(' - ', '-') for i in predicted_relations]
            predictions.append(predicted_relations)
            
    # for evaluating NER and RE separately
    def split_string(string):
        data = {}
        substrings = string.split('@NER@')
        data['NER'] = substrings[0].split(';')
        # remove unexpected spaces around '-'
        data['NER'] = [i.replace(' - ', '-').lower() for i in data['NER']]
        data['NER'] = tuple([item.strip() for item in data['NER']])
        data['REL'] = substrings[1].replace(';','@DRUG@')
        data['REL'] = data['REL'].replace('@POS@', '@DRUG@ @POS@')
        data['REL'] = data['REL'].replace('@COMB@', '@DRUG@ @COMB@')
        return(data)
      
    # initialization
    true_positive_sum, pred_sum, true_sum = 0, 0, 0
    ner_true_positive_sum, ner_pred_sum, ner_true_sum = 0, 0, 0
    
    # counting true_positive_sum, pred_sum and true_sum for NER
    for i in range(len(gold)):
        pre = predictions[i][0]
        gol = gold[i]
        pre = split_string(pre)
        gol = split_string(gol)
        p = {}
        g = {}
        for j in set(pre['NER']):
            p[j] = pre['NER'].count(j)
        for j in set(gol['NER']):
            g[j] = gol['NER'].count(j)
            
        # find the intersection of predicted entities and gold entities
        for k in set(p.keys()).intersection(set(g.keys())):
            # count the smaller one as true_positive_sum
            ner_true_positive_sum += min(p[k], g[k])

        ner_pred_sum += len(pre['NER'])
        ner_true_sum += len(gol['NER'])
        
        # counting positive_sum, pred_sum and true_sum for RE
        gold_annotations = util.extract_relations([gol['REL']], remove_duplicate_ents=True)
        pred_annotations = util.extract_relations([pre['REL']], remove_duplicate_ents=True)
        for pred_ann, gold_ann in zip(pred_annotations, gold_annotations):
            pred_rels = pred_ann.get('POS', [])
            dedup_pred_rels = set(pred_rels)
            pred_sum += len(dedup_pred_rels)
            if args.metric == 'any_combination_f1':
                pred_rels = pred_ann.get('COMB', [])
                dedup_pred_rels = set(pred_rels)
                pred_sum += len(dedup_pred_rels)
            if gold_ann:
                if args.metric == 'positive_combination_f1':
                    gold_rels = gold_ann.get('POS', [])
                    pred_rels = pred_ann.get('POS', [])
                if args.metric == 'any_combination_f1':
                    gold_rels = gold_ann.get('POS', []) + gold_ann.get('COMB', [])
                    pred_rels = pred_ann.get('POS', []) + pred_ann.get('COMB', [])
                # convert to a set, as we don't care about duplicates or order.
                dedup_pred_rels = set(pred_rels)
                dedup_gold_rels = set(gold_rels)
                true_positive_sum += len(  # type: ignore
                    dedup_pred_rels & dedup_gold_rels
                )
                true_sum += len(dedup_gold_rels)

    R = true_positive_sum/true_sum
    P = true_positive_sum/pred_sum
    Fscore = 2 * P * R / (P + R)

    ner_R = ner_true_positive_sum / ner_true_sum
    ner_P = ner_true_positive_sum / ner_pred_sum
    ner_Fscore = 2 * ner_P * ner_R / (ner_P + ner_R)

    print('NER Recall: ', ner_R)
    print('NER Precision: ', ner_P)
    print('NER F1: ', ner_Fscore)

    print(args.metric + ':')
    print('Recall: ', R)
    print('Precision: ', P)
    print('F1: ', Fscore)

# do positive combination evaluation
if not args.ner and args.metric == 'positive_combination_f1':
    # initialization
    true_positive_sum, pred_sum, true_sum = 0, 0, 0
    predictions=[]
    gold=[]
    # counting positive_sum, pred_sum and true_sum
    with open(cached_path('n-ary/test.txt'), "r") as data_file:
        for line_num, line in enumerate(data_file):
            line = line.strip("\n")
            line_parts = line.split('\t')
            input_text = line_parts[0]
            gold.append(line_parts[1])
            gold_relations = [line_parts[1]]
            predicted_relations = seq2rel(input_text)
            # remove unexpected spaces around '-'
            predicted_relations = [i.replace(' - ', '-') for i in predicted_relations]
            predictions.append(predicted_relations)
            gold_annotations = util.extract_relations(gold_relations, remove_duplicate_ents=True)
            pred_annotations = util.extract_relations(predicted_relations, remove_duplicate_ents=True)
            for pred_ann, gold_ann in zip(pred_annotations, gold_annotations):
                pred_rels = pred_ann.get('POS', [])
                dedup_pred_rels = set(pred_rels)
                pred_sum += len(dedup_pred_rels)
                if gold_ann:
                    for rel_label, gold_rels in gold_ann.items():
                        pred_rels = pred_ann.get(rel_label, [])
                        # convert to a set, as we don't care about duplicates or order.
                        dedup_pred_rels = set(pred_rels)
                        dedup_gold_rels = set(gold_rels)
                        if rel_label == 'POS':
                            true_positive_sum += len(  # type: ignore
                                dedup_pred_rels & dedup_gold_rels
                            )
                            true_sum += len(dedup_gold_rels)

    R = true_positive_sum/true_sum
    P = true_positive_sum/pred_sum
    Fscore = 2 * P * R / (P + R)

    print(args.metric + ':')
    print('Recall: ', R)
    print('Precision: ', P)
    print('F1: ', Fscore)

# do any combination 3-way evaluation
if not args.ner and args.metric == 'any_combination_f1':
    # initialization
    true_positive_sum, pred_sum, true_sum = 0, 0, 0
    predictions=[]
    gold=[]
    # counting positive_sum, pred_sum and true_sum
    with open(cached_path('n-ary-fixed-order/test.txt'), "r") as data_file:
        for line_num, line in enumerate(data_file):
            line = line.strip("\n")
            line_parts = line.split('\t')
            input_text = line_parts[0]
            gold.append(line_parts[1])
            gold_relations = [line_parts[1]]
            predicted_relations = seq2rel(input_text)
            # remove unexpected spaces around '-'
            predicted_relations = [i.replace(' - ', '-') for i in predicted_relations]
            predictions.append(predicted_relations)
            gold_annotations = util.extract_relations(gold_relations, remove_duplicate_ents=True)
            pred_annotations = util.extract_relations(predicted_relations, remove_duplicate_ents=True)
            for pred_ann, gold_ann in zip(pred_annotations, gold_annotations):
                pred_rels = pred_ann.get('POS', [])
                dedup_pred_rels = set(pred_rels)
                pred_sum += len(dedup_pred_rels)
                pred_rels = pred_ann.get('COMB', [])
                dedup_pred_rels = set(pred_rels)
                pred_sum += len(dedup_pred_rels)
                if gold_ann:
                    gold_rels = gold_ann.get('POS', []) + gold_ann.get('COMB', [])
                    pred_rels = pred_ann.get('POS', []) + pred_ann.get('COMB', [])
                    # convert to a set, as we don't care about duplicates or order.
                    dedup_pred_rels = set(pred_rels)
                    dedup_gold_rels = set(gold_rels)
                    true_positive_sum += len(  # type: ignore
                        dedup_pred_rels & dedup_gold_rels
                    )
                    true_sum += len(dedup_gold_rels)

    R = true_positive_sum/true_sum
    P = true_positive_sum/pred_sum
    Fscore = 2 * P * R / (P + R)

    print(args.metric + ':')
    print('Recall: ', R)
    print('Precision: ', P)
    print('F1: ', Fscore)
