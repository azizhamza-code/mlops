from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os



class encoder_(object):

    def __init__(self,classes=None):

        self.mlb = None
        self.trained = False
        if classes is not None:
            self.classes = classes
            self.mlb = MultiLabelBinarizer(classes=self.classes)

    def fit(self,y_train):
        self.mlb =self.mlb.fit(y_train)
        self.trained = True

    def encode(self,y):
        if self.trained :
            y = self.mlb.transform(y)
            return y
        # TODO: make it log error
        print("not yet fited")

    def decode(self,y_encoded):
         if self.trained :
            y = self.mlb.inverse_transform(y_encoded)
            return y
         # TODO: make it log error
         print("not yet fited")

    def save(self,target_dir):
        if self.trained :
            file_path = os.path.join(target_dir , 'encoder.gz')
            joblib.dump(self.mlb , file_path)
        # TODO: make it log error
        print("not yet fited")

    def load(self,target_dir):
        file_path = os.path.join(target_dir , 'encoder.gz')
        self.trained = True
        self.mlb = joblib.load(file_path)

class extract(object):

    def __init__(self,classes=None):

        self.exractor =  TfidfVectorizer()
        self.trained = False

    def fit(self,x_train):
        self.exractor = self.exractor.fit(x_train)
        self.trained = True

    def encode(self,x_val):
        if self.trained :
            x_val = self.exractor.transform(x_val)
            return x_val
        # TODO: make it log error
        print("not yet fited")

    def save(self,target_dir):
        if self.trained :
            file_path = os.path.join(target_dir , 'extractor.gz')
            joblib.dump(self.exractor , file_path)
        # TODO: make it log error
        print("not yet fited")

    def load(self,target_dir):
        file_path = os.path.join(target_dir , 'extractor.gz')
        self.trained = True
        self.exractor = joblib.load(file_path)


        
