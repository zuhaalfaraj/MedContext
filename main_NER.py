import train_ner
from help import load_data_spacy
import spacy
from spacy import displacy


def load_model(model_path):
    ''' Loads a pre-trained model for prediction on new test sentences

    model_path : directory of model saved by spacy.to_disk
    '''
    nlp = spacy.blank('en')
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    ner = nlp.from_disk(model_path)
    return ner

if __name__ == '__main__':
    TRAIN_DATA, LABELS = load_data_spacy("NERdata/BC5CDR-disease/train.tsv")
    TEST_DATA, _ = load_data_spacy("NERdata/BC5CDR-disease/test.tsv")
    VALID_DATA, _ = load_data_spacy("NERdata/BC5CDR-disease/train_dev.tsv")

    print(TRAIN_DATA)
    ner_trainer = train_ner.Train_NER(TRAIN_DATA, TEST_DATA, VALID_DATA)

    nlp, valid_f1scores, test_f1scores= ner_trainer.train_spacy(TRAIN_DATA,LABELS,30)
    #nlp.to_disk('model')

    #nlp = load_model('model')
    doc= nlp(TEST_DATA[0][0])
    for ent in doc.ents:
        print(ent.text, ent.label_)



    """


