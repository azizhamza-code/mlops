from mzops.utils import set_seeds
from mzops.config import d
from mzops.tag_predict import predict 
from mzops.train  import train 


set_seeds()

if __name__ == "__main__":

    if d['status'] == 'predict':
        predict(d['sample'])
    else:
        train()




