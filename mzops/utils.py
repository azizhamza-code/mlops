
from typing import List
from pathlib import Path
import os
from tqdm import tqdm
import requests
from ast import literal_eval
import pandas as pd

file_path = Path(__file__).resolve().parent
GITHUB_URL = "https://github.com/azizhamza-code/NLP/"


def download_file(url: str, file_path: str) -> None:
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-lenght', 0))
    try:
        with open(file_path, 'wb', buffering=16*1024*1024) as f:
            bar = tqdm(total=total_size, unit='B', unit_scale=True)
            bar.set_description(os.path.split(file_path)[-1])
            for chunk in response.iter_content(32*1024):
                f.write(chunk)
                bar.update(len(chunk))
            bar.close()
    except Exception:
        print("download failed")


def download_from_github(target_dir: str, file: str, version, force: False):
    url = GITHUB_URL + f"releases/download/{version}/{file}"
    file_path = os.path.join(target_dir, file)
    if os.path.exists(file_path) and not force:
        print(f"{file} already downloaded in {file_path}")
        return
    download_file(url, file_path)


def import_successive(target_dir: str, files: List[str], version: str, force: bool) -> None:
    os.makedirs(target_dir, exist_ok=True)
    for file in files:
        download_from_github(target_dir, file, version, force=False)


def import_data(target_dir: str = "data", force: bool = False) -> None:

    import_successive(target_dir=target_dir, force=force, files=[
        "train.tsv", "test.tsv", "validation.tsv", "text_prepare_tests.tsv"],
        version="datav1")

def read_data(file:str,test:bool = False):
    data = pd.read_csv(file,sep='\t')
    if not test:
        data['tags'] = data['tags'].apply(literal_eval)
    return data


    
