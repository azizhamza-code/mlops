from pandas import DataFrame
from nltk.corpus import stopwords
import re
from mzops.utils import import_data , read_data , data_dir
import nltk
import os
nltk.download('stopwords')



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
#TODO: make it on config file  
#TODO: make it os independent

data_dir = data_dir()

def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub(BAD_SYMBOLS_RE, "", text)
    text = re.sub(' +', ' ', text)
    # delete stopwords from text
    text = [word for word in text.split(" ") if word not in STOPWORDS]
    return " ".join(text)


def spli_x_y(train ,val ,test):

    x_train  , y_train = train['title'].values , train['tags'].values
    x_val , y_val = val['title'].values , val['tags'].values
    x_test = test['title'].values

    return x_train ,y_train , x_val ,y_val ,x_test

def etl(dev_mode):

    import_data(target_dir=data_dir)
    train, val, test = read_data("train.tsv",target_dir = data_dir,dev_mode = dev_mode), read_data(
        "validation.tsv",target_dir = data_dir,dev_mode = dev_mode), read_data("test.tsv",target_dir = data_dir, test=True,dev_mode = dev_mode)

    x_train , y_train , x_val , y_val , x_test =spli_x_y(train , val ,test)

    x_train = [text_prepare(x) for x in x_train]
    x_val = [text_prepare(x) for x in x_val]
    x_test = [text_prepare(x) for x in x_test]
        
    return x_train , y_train , x_val , y_val , x_test








