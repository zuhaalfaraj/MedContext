import spacy
import random
import time
import numpy as np
from spacy.util import minibatch, compounding
import sys
from spacy import displacy
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_data_spacy(file_path):
    ''' Converts data from:
    word \t label \n word \t label \n \n word \t label
    to: sentence, {entities : [(start, end, label), (stard, end, label)]}
    '''
    file = open(file_path, 'r')
    training_data, entities, sentence, unique_labels = [], [], [], []
    current_annotation = None
    start = 0
    end = 0  # initialize counter to keep track of start and end characters
    for line in file:
        line = line.strip("\n").split("\t")
        # lines with len > 1 are words
        if len(line) > 1:
            label = line[1]
            if (label != 'O'):
                label = line[1] + "_Disease"  # the .txt is formatted: label \t word, label[0:2] = label_type
            # label_type = line[0][0] # beginning of annotations - "B", intermediate - "I"
            word = line[0]
            sentence.append(word)
            start = end
            end += (len(word) + 1)  # length of the word + trailing space

            if label == 'I_Disease':  # if at the end of an annotation
                entities.append((start, end - 1, label))  # append the annotation

            if label == 'B_Disease':  # if beginning new annotation
                entities.append((start, end - 1, label))  # start annotation at beginning of word

            if label != 'O' and label not in unique_labels:
                unique_labels.append(label)

        # lines with len == 1 are breaks between sentences
        if len(line) == 1:
            if (len(entities) > 0):
                sentence = " ".join(sentence)
                training_data.append([sentence, {'entities': entities}])

            # reset the counters and temporary lists
            end = 0
            start = 0
            entities, sentence = [], []
    file.close()
    return training_data, unique_labels

def calc_precision(pred, true):
    precision = len([x for x in pred if x in true]) / (len(pred) + 1e-20) # true positives / total pred
    return precision

def calc_recall(pred, true):
    recall = len([x for x in true if x in pred]) / (len(true) + 1e-20)    # true positives / total test
    return recall

def calc_f1(precision, recall):
    f1 = 2 * ((precision * recall) / (precision + recall + 1e-20))
    return f1

def evaluate(ner, data ):
    preds = [ner(x[0]) for x in data]
    precisions, recalls, f1s = [], [], []

    for pred, true in zip(preds, data):
        true = [x[2] for x in list(chain.from_iterable(true[1].values()))] # x[2] = annotation, true[1] = (start, end, annot)
        pred = [i.label_ for i in pred.ents] # i.label_ = annotation label, pred.ents = list of annotations
        precision = calc_precision(true, pred)
        precisions.append(precision)
        recall = calc_recall(true, pred)
        recalls.append(recall)
        f1s.append(calc_f1(precision, recall))

    return {"textcat_p": np.mean(precisions), "textcat_r": np.mean(recalls), "textcat_f":np.mean(f1s)}



if __name__=='__main__':

    TRAIN_DATA, LABELS = load_data_spacy("NERdata/BC5CDR-disease/train.tsv")
    print(LABELS)
    #print(f.read())
    """    
    TRAIN_DATA, LABELS = load_data_spacy("NERdata/BC5CDR-disease/train.tsv")
    print(TRAIN_DATA)
    print(len(TRAIN_DATA))
    TEST_DATA, _ = load_data_spacy("NERdata/BC5CDR-disease/test.tsv")
    print(len(TEST_DATA))
    VALID_DATA, _ = load_data_spacy("NERdata/BC5CDR-disease/train_dev.tsv")
    print(len(VALID_DATA))

    """
