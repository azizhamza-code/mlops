from mzops.model import encoder_ , extract
from mzops.config import d
import joblib
from mzops.data import text_prepare
from mzops.utils import set_logger_module

logger = set_logger_module(__name__)

def predict(title:str):
    logger.info(" start predction!!!")
    title =[title]
    title = ([text_prepare(title) for title in title])
    encoder = encoder_()
    encoder.load(d['artifactdir'])
    extractor = extract()
    extractor.load(d['artifactdir'])
    model = joblib.load(d['model_artifact'])


    x_vec = extractor.encode(title)
    y_encoded = model.predict(x_vec)
    y_decoded = encoder.decode(y_encoded)

    print(f'the tags of {title} is {y_decoded}')

