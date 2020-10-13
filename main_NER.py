import train_ner
from help import load_data_spacy
import spacy
from spacy import displacy
import cv2
import pytesseract
import os

class NameEntitiyRecognitionClinicla:

    def new_model(self,TRAIN_DATA, LABELS, iter):
        ner_trainer = train_ner.Train_NER(TRAIN_DATA, TEST_DATA, VALID_DATA)
        nlp, valid_f1scores, test_f1scores = ner_trainer.train_model(LABELS, iter)
        return nlp, valid_f1scores, test_f1scores
    def load_model(self,model_path):

        nlp = spacy.blank('en')
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner)
        ner = nlp.from_disk(model_path)
        return ner

    def get_entities(self,model_path,text):
        ner=self.load_model(model_path)
        test_sentences = [x[0] for x in text]
        print(test_sentences[0])
        for x in test_sentences:
            #print(x)
            doc = ner(x)
            for i in doc.ents:
                print(i, i.label_)
            displacy.render(doc, jupyter=False, style="ent")

    def load_text(self,file_path):

        file = open(file_path, 'r')
        sentence,data = [],[]
        end = 0  # initialize counter to keep track of start and end characters
        for line in file:
            if not line.isspace():
                print(line)
                line = line.strip("\n").split("\t")

                sentence.append(line)


        return sentence

    def img_to_txt(self, img_path):
        file_name= os.path.join('texts',img_path.split('.')[0]+".txt")
        file = open(file_name, "w+")
        file.write("")

        img = cv2.imread(img_path)
        text = pytesseract.image_to_string(img)

        file.write(text)
        file.close()

        return file_name

    def full_process(self,img_path):
        text_path= self.img_to_txt(img_path)
        data= self.load_text(text_path)

        return data

if __name__ == '__main__':
    TRAIN_DATA, LABELS = load_data_spacy("NERdata/BC5CDR-disease/train.tsv")
    TEST_DATA, _ = load_data_spacy("NERdata/BC5CDR-disease/test.tsv")
    VALID_DATA, _ = load_data_spacy("NERdata/BC5CDR-disease/train_dev.tsv")

    main= NameEntitiyRecognitionClinicla()
    text= main.full_process('sr.jpeg')
    print(text)
    #text= main.img_to_txt("medical-report.jpg")
    main.get_entities('model',text)

    #print(TEST_DATA[0:15])




