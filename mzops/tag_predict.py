from mzops.model import encoder_ , extract
from mzops.config import d
import joblib
from mzops.data import text_prepare



def predict(title:str):
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

