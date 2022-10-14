import logging
import os
from pathlib import Path
import sys
from rich.logging import RichHandler

BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = Path(BASE_DIR, "data")
ARTIFACT_DIR  = Path(BASE_DIR,'artifact')
LOG_DIR = Path(BASE_DIR,"logs")

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


#logger configuration

def get_log_handlers():
    """ different handlers depend on severity of log

    config two logger handler:

    c_handler : take charge of DEBUG ,INFO and WARNING log and print them on the terminal 
    f_handler : take charge of ERROR and CRITICAL severity log and print them on file 

    Returns:
        dict: handlers
    """
    #c_handler = logging.StreamHandler(sys.stdout)
    c_handler = RichHandler(markup=True)  
    f_handler = logging.FileHandler(Path(LOG_DIR,'log.log'),mode='w')
    f_handler.setLevel(logging.ERROR)
    c_handler.setLevel(logging.DEBUG)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    c_handler.setFormatter(c_format)
    return {'stdout':c_handler,'file_log':f_handler}

d = {}
d['status'] = 'predict'
d['sub_data'] = True
d['sample'] = 'the best  os machine windows python javascript and pandas numpy'
d['artifactdir'] = ARTIFACT_DIR
d['model_artifact'] = Path(ARTIFACT_DIR, 'model.gz')
