from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer , HashingVectorizer
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

    def fit(self,y_train , what:str = 'train'):
        self.mlb =self.mlb.fit(y_train)
        self.trained = True
        logger.info(f"encoder fitted {what}")

    def encode(self,y,what:str='train'):
        if self.trained :
            y = self.mlb.transform(y)
            logger.info(f"label encoded {what}")
            return y
            
        else:
            logger.error("encoder not yet trained")

    def decode(self,y_encoded ,what:str = 'train'):
        if self.trained :
            y = self.mlb.inverse_transform(y_encoded)
            logger.info(f"label decoded {what}")
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



class hash_extractor(object):

    def __init__(self,num_features=None):

        num_features = num_features if not None else (2 ** 18)
        self.exractor =   HashingVectorizer(decode_error='ignore', n_features=num_features) 
        self.trained = False

    def fit(self,x_train):
        self.exractor = self.exractor.fit(x_train)
        self.trained = True
        logger.info("exractor fitted")

    def encode(self,x_val,what:str = 'train'):
        if self.trained :
            x_val = self.exractor.transform(x_val)
            logger.info(f"text vectorized {what}")
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

class tfidf_extractor(object):

    def __init__(self):

        self.exractor =  TfidfVectorizer()
        self.trained = False

    def fit(self,x_train):
        self.exractor = self.exractor.fit(x_train)
        self.trained = True
        logger.info("exractor fitted")

    def encode(self,x_val,what:str = 'train'):
        if self.trained :
            x_val = self.exractor.transform(x_val)
            logger.info(f"text vectorized {what}")
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