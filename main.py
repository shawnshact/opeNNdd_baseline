import os
import sys
from baseline import Baseline

db_path = str(sys.argv[1])
storage_folder = str(sys.argv[2])
BATCH_SIZE = 10
CHUNK_SIZE = 1000

model = Baseline(db_path, storage_folder, BATCH_SIZE)
model.train()
model.db_file.close()
