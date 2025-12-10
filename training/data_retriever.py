import gzip
import os
import shutil
import urllib.request

url = "https://storage.googleapis.com/chesspic/datasets/2021-07-31-lichess-evaluations-37MM.db.gz"
gz_path = "2021-07-31-lichess-evaluations-37MM.db.gz"
db_path = "2021-07-31-lichess-evaluations-37MM.db"

urllib.request.urlretrieve(url, gz_path)

with gzip.open(gz_path, 'rb') as f_in:
    with open(db_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

os.remove(gz_path)