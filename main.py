import os
import sys
from baseline import Baseline
import baseline_utils as utils

db_path = str(sys.argv[1])
storage_folder = str(sys.argv[2])
BATCH_SIZE = 10
CHUNK_SIZE = 1000

model = Baseline(db_path, storage_folder, BATCH_SIZE)
utils.train(model)
model.db_file.close()
