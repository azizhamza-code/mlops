from pandas import DataFrame
from nltk.corpus import stopwords
import re
from mzops.utils import import_data, read_data
from mzops.utils import stopswords_setup
import os
from mzops.config import DATA_DIR, get_log_handlers
import logging
from mzops.utils import set_logger_module

stopswords_setup()

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

logger = set_logger_module(__name__)

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


def spli_x_y(train, val, test):

    logger.info("split start")

    x_train, y_train = train['title'].values, train['tags'].values
    x_val, y_val = val['title'].values, val['tags'].values
    x_test = test['title'].values

    logger.info("split done")
    
    return x_train, y_train, x_val, y_val, x_test


def etl(dev_mode):
    logger.info("etl start")

    import_data(target_dir=DATA_DIR)
    train, val, test = read_data("train.tsv", target_dir=DATA_DIR, dev_mode=dev_mode), read_data(
        "validation.tsv", target_dir=DATA_DIR, dev_mode=dev_mode), read_data("test.tsv", target_dir=DATA_DIR, test=True, dev_mode=dev_mode)

    x_train, y_train, x_val, y_val, x_test = spli_x_y(train, val, test)

    x_train = [text_prepare(x) for x in x_train]
    x_val = [text_prepare(x) for x in x_val]
    x_test = [text_prepare(x) for x in x_test]
    logger.info("etl done")

    return x_train, y_train, x_val, y_val, x_test
