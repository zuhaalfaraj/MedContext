
from spacy.util import minibatch, compounding
import spacy
import random
from help import  evaluate
import os
class Train_NER:
    def __init__(self,TRAIN_DATA,TEST_DATA,VALID_DATA,):
        self.TEST_DATA=TEST_DATA
        self.VALID_DATA=VALID_DATA
        self.TRAIN_DATA=TRAIN_DATA

    def train_model(self, labels, iterations, dropout=0.5, display_freq=1):
        nlp, valid_f1scores, test_f1scores=self.train_spacy(self.TRAIN_DATA, labels, iterations, dropout, display_freq )
        return nlp, valid_f1scores, test_f1scores

    def train_spacy(self,train_data, labels, iterations, model_name,dropout=0.5, display_freq=1):
        '''
         Train a spacy NER model, which can be queried against with test data

        train_data : training data in the format of (sentence, {entities: [(start, end, label)]})
        labels : a list of unique annotations
        iterations : number of training iterations
        dropout : dropout proportion for training
        display_freq : number of epochs between logging losses to console
        '''
        max_f2a= 0
        valid_f1scores = []
        test_f1scores = []
        os.mkdir(os.path.join('models',model_name))
        nlp = spacy.load("en_core_web_md")
        # nlp = spacy.blank('en')
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner)
        else:
            ner = nlp.get_pipe("ner")

        # Add entity labels to the NER pipeline
        for i in labels:
            ner.add_label(i)

        # Disable other pipelines in SpaCy to only train NER
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):
            # nlp.vocab.vectors.name = 'spacy_model' # without this, spaCy throws an "unnamed" error
            optimizer = nlp.begin_training()
            for itr in range(iterations):
                random.shuffle(train_data)  # shuffle the training data before each iteration
                losses = {}
                batches = minibatch(train_data, size=compounding(16.0, 64.0, 1.5))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts,
                        annotations,
                        drop=dropout,
                        sgd=optimizer,
                        losses=losses)
                # if itr % display_freq == 0:
                #    print("Iteration {} Loss: {}".format(itr + 1, losses))

                scores = evaluate(nlp, self.VALID_DATA)
                valid_f1scores.append(scores["textcat_f"])
                print('=======================================')
                print('Interation = ' + str(itr))
                print('Losses = ' + str(losses))
                print('===============VALID DATA========================')

                print('F1-score = ' + str(scores["textcat_f"]))
                print('Precision = ' + str(scores["textcat_p"]))
                print('Recall = ' + str(scores["textcat_r"]))
                scores = evaluate(nlp, self.TEST_DATA)
                test_f1scores.append(scores["textcat_f"])
                print('===============TEST DATA========================')
                print('F1-score = ' + str(scores["textcat_f"]))
                print('Precision = ' + str(scores["textcat_p"]))
                print('Recall = ' + str(scores["textcat_r"]))
                print('=======================================')

                if scores["textcat_f"] > max_f2a:
                    max_f2a = scores["textcat_f"]
                    nlp.to_disk(os.path.join('models',model_name))
                    print('Model of {} f1 has saved'.format(max_f2a))
        return nlp, valid_f1scores, test_f1scores


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_md")
    text = "Google has a great search engine that was invented by John Smith"
    doc = nlp(text)
    print(doc.ents)
    for ent in doc.ents:
        print(ent.text, ent.label_)