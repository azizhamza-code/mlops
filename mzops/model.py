from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from mzops.utils import set_logger_module


logger = set_logger_module(__name__)
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
        logger.info("encoder fitted")

    def encode(self,y):
        if self.trained :
            y = self.mlb.transform(y)
            logger.info("label encoded ")
            return y
            
        else:
            logger.error("encoder not yet trained")

    def decode(self,y_encoded):
        if self.trained :
            y = self.mlb.inverse_transform(y_encoded)
            logger.info("label decoded ")
            return y
        else:
            logger.error("encoder not yet trained")

    def save(self,target_dir):
        if self.trained :
            file_path = os.path.join(target_dir , 'encoder.gz')
            joblib.dump(self.mlb , file_path)
            logger.info(f"encoder saved at{file_path} ")
        else:
            logger.error("encoder not yet trained")

    def load(self,target_dir):
        file_path = os.path.join(target_dir , 'encoder.gz')
        self.trained = True
        self.mlb = joblib.load(file_path)
        logger.info(f"encoder loaded from")

class extract(object):

    def __init__(self,classes=None):

        self.exractor =  TfidfVectorizer()
        self.trained = False

    def fit(self,x_train):
        self.exractor = self.exractor.fit(x_train)
        self.trained = True
        logger.info("exractor fitted")

    def encode(self,x_val):
        if self.trained :
            x_val = self.exractor.transform(x_val)
            logger.info("text vectorized")
            return x_val
        else:
            logger.error("encoder not yet trained")

    def save(self,target_dir):
        if self.trained :
            file_path = os.path.join(target_dir , 'extractor.gz')
            joblib.dump(self.exractor , file_path)
            logger.info("exractor saved")
        else:
            logger.error("encoder not yet trained")

    def load(self,target_dir):
        file_path = os.path.join(target_dir , 'extractor.gz')
        self.trained = True
        self.exractor = joblib.load(file_path)
        logger.info("extractor loaded")


        
