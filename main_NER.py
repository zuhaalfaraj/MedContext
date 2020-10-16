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
        test_sentences = [x[0] for x in text[:4]]
        all_ents=[]
        for x in text:
            #print(x)
            doc = ner(x)
            all_ents.append(doc.ents)
        all_ents= self.arrange_output(all_ents)

        return all_ents



    def load_text(self,file_path):

        file = open(file_path, 'r')
        sentence,data = "",[]
        end = 0  # initialize counter to keep track of start and end characters
        for line in file:
            if not line.isspace():
                print(line)
                line = line.strip("\n").split("\t")

                sentence+=" "+str(line)


        return sentence

    def arrange_output(self, out):
        all_out=[]
        print(out)
        for ent in out:
            word=None
            for x in ent:
                if x.label_[0] == "B":
                    if word!=None:
                        all_out.append(word)
                    word=str(x)
                elif x.label_[0] == "I":
                    word+= " "+str(x)
        all_out.append(word)
        return all_out

    def img_to_txt(self, img_path):
        img = cv2.imread(img_path)
        text = pytesseract.image_to_string(img)
        out = text.strip("\n").split("\t")
        return out

    def full_process(self,img_path):
        text_path= self.img_to_txt(img_path)
        data= self.load_text(text_path)

        return data

if __name__ == '__main__':
    TRAIN_DATA, LABELS = load_data_spacy("NERdata/BC5CDR-disease/train.tsv")
    TEST_DATA, _ = load_data_spacy("NERdata/BC5CDR-disease/test.tsv")
    VALID_DATA, _ = load_data_spacy("NERdata/BC5CDR-disease/train_dev.tsv")

    main= NameEntitiyRecognitionClinicla()
    #text= main.full_process('list.png')

    text= main.img_to_txt("medical-report.jpg")
    print(text)
    #print(main.get_entities('model',TEST_DATA))





