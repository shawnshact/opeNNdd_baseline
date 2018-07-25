import os
import sys
from multiprocessing import Process, Queue, Pipe
import numpy as np
import h5py as h5
import itertools
import random
import math
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from datetime import datetime
random.seed(datetime.now())

def binSearch(arr, target):
    low = 0
    high = len(arr)
    mid = (low + high)//2
    found = False

    if target < arr[0]:
        return 0

    while (not found):
        if (target < arr[mid] and target >=arr[mid-1]):
            found = True
        elif (target >= arr[mid]):
            low = mid+1
            mid = (low+high)//2
        else:
            high = mid-1
            mid = (low+high)//2
    return mid

class Baseline:
    train_split, val_split, test_split = .7,.1,.2
    grid_dim, channels = 72, 6
    chunk_size = 50
    def __init__(self,db_path=None, storage_folder=None, batch_size=None, id=None):

        if (id):
            assert (os.path.isdir(os.path.join(storage_folder, 'tmp', str(id)))), "Unable to locate model " + str(id) + " in the specified storage folder" + storage_folder
            self.id = id
            self.existing_model = True
        else:
            self.id = random.randint(100000, 999999) #generate random 6-digit model id
            self.existing_model = False

        self.batch_size = batch_size
        assert os.path.isfile(db_path), "Database does not exist in specified location: " + str(db_path)

        None if os.path.isdir(storage_folder) else os.makedirs(storage_folder) #create dir if need be
        None if os.path.isdir(os.path.join(storage_folder, 'tmp')) else os.makedirs(os.path.join(storage_folder, 'tmp'))
        None if os.path.isdir(os.path.join(storage_folder, 'logs')) else os.makedirs(os.path.join(storage_folder, 'logs'))
        self.storage_folder = storage_folder #storage folder for model and logs
        self.model_folder = os.path.join(storage_folder, 'tmp', str(self.id))
        self.log_folder = os.path.join(storage_folder, 'logs', str(self.id))



        self.model_file = os.path.join(self.model_folder,str(self.id)+'.pkl')

        self.db_file = h5.File(db_path, mode='r') #handle to file
        self.chunk_names = [name for name in self.db_file['labels']]
        self.data_chunks = [len(self.db_file['labels'][partition]) for partition in self.db_file['labels']]
        self.chunk_thresholds = list(itertools.accumulate(self.data_chunks))
        self.total_members = self.chunk_thresholds[-1]

        self.train_members = int(round(self.train_split*self.total_members))
        self.val_members = int(round(self.val_split*self.total_members))
        self.test_members = int(round(self.test_split*self.total_members))

        self.total_train_chunks = int(math.ceil(self.train_members/self.chunk_size))
        self.total_val_chunks = int(math.ceil(self.val_members/self.chunk_size))

        self.train_steps = int(math.ceil(self.chunk_size/batch_size))
        self.val_steps = int(math.ceil(self.chunk_size/batch_size))

        member_indices = list(range(self.total_members))
        random.shuffle(member_indices)

        self.train_indices = member_indices[:self.train_members]
        self.val_indices = member_indices[self.train_members:self.train_members+self.val_members]
        self.test_indices = member_indices[-self.test_members:]

        self.train_db_index, self.val_db_index,self.test_db_index = 0,0,0
        self.train_chunk_index, self.val_chunk_index = 0,0
        self.epochs, self.optimal_epochs = 0, 0
        self.min_epochs, self.stop_threshold = 0, 5
        self.master, self.subprocess = Pipe()
        self.running_process = False
        self.train_queue = Queue()
        self.val_queue = Queue()

        self.max_queue_size = 5
        #self.db_file.close()

    def shuffle_train_data(self):
        random.shuffle(self.train_indices)
        self.train_db_index = 0

    def shuffle_val_data(self):
        random.shuffle(self.val_indices)
        self.val_db_index = 0

    def shuffle_test_data(self):
        random.shuffle(self.test_indices)
        self.test_db_index = 0

    def next_train_chunk(self, chunk_index):
        flag = False
        chunk_size = self.chunk_size
        #get the next batch
        if (self.train_members - self.train_db_index) < chunk_size:
            flag = True
            chunk_size = self.train_members%chunk_size

        batch_ligands = np.zeros([chunk_size, self.grid_dim*self.grid_dim*self.grid_dim*self.channels], dtype=np.float32)
        batch_energies = np.zeros([chunk_size], dtype=np.float32)
        for i in range(self.train_db_index, self.train_db_index+chunk_size):
            file_index = binSearch(self.chunk_thresholds, self.train_indices[i])
            filename = str(self.chunk_names[file_index])
            chunk_index = (self.chunk_thresholds[file_index]-self.chunk_thresholds[file_index-1]-1) if file_index > 0 else self.train_indices[i]
            batch_ligands[i-self.train_db_index] = self.db_file['ligands'][filename][chunk_index].flatten()
            batch_energies[i-self.train_db_index] = self.db_file['labels'][filename][chunk_index]

        if flag:
            self.train_db_index = 0
        else:
            self.train_db_index += chunk_size

        #return as np arrays
        self.temp_chunk_size = chunk_size
        self.train_queue.put([batch_ligands, batch_energies])
        self.subprocess.send(False)
        chunk_index = self.train_db_index

    def next_val_chunk(self):
        flag = False
        chunk_size = self.chunk_size
        #get the next batch
        if (self.val_members - self.val_db_index) < chunk_size:
            flag = True
            chunk_size = self.val_members%chunk_size

        batch_ligands = np.zeros([chunk_size, self.grid_dim*self.grid_dim*self.grid_dim*self.channels], dtype=np.float32)
        batch_energies = np.zeros([chunk_size], dtype=np.float32)
        for i in range(self.val_db_index, self.val_db_index+chunk_size):
            file_index = binSearch(self.chunk_thresholds, self.val_indices[i])
            filename = str(self.chunk_names[file_index])
            chunk_index = (self.chunk_thresholds[file_index]-self.chunk_thresholds[file_index-1]-1) if file_index > 0 else self.val_indices[i]
            batch_ligands[i-self.val_db_index] = self.db_file['ligands'][filename][chunk_index].flatten()
            batch_energies[i-self.val_db_index] = self.db_file['labels'][filename][chunk_index]

        if flag:
            self.val_db_index = 0
        else:
            self.val_db_index += chunk_size

        self.temp_chunk_size = chunk_size
        #return as np arrays
        self.val_queue.put([batch_ligands, batch_energies])
        self.subprocess.send(False)
        chunk_index = self.val_db_index

    def next_train_batch(self, chunk_size):
        flag = False
        chunk_index = self.train_chunk_index
        batch_size = self.batch_size

        if (chunk_index + batch_size) > chunk_size:
            flag = True
            batch_size = chunk_size%batch_size

        batch_ligands = self.train_receiver[0][chunk_index:chunk_index+batch_size]
        batch_labels = self.train_receiver[1][chunk_index:chunk_index+batch_size]

        batch_labels = self.train_receiver[1][chunk_index:chunk_index+batch_size]

        if flag:
            chunk_index = 0
        else:
            chunk_index +=batch_size

        self.train_chunk_index = chunk_index

        return batch_ligands, batch_labels

    def next_val_batch(self, chunk_size):
        flag = False
        chunk_index = self.val_chunk_index
        batch_size = self.batch_size

        if (chunk_index + batch_size) > chunk_size:
            flag = True
            batch_size = chunk_size%batch_size

        batch_ligands = self.val_receiver[0][chunk_index:chunk_index+batch_size]
        batch_labels = self.val_receiver[1][chunk_index:chunk_index+batch_size]

        if flag:
            chunk_index = 0
        else:
            chunk_index +=batch_size

        self.val_chunk_index = chunk_index

        return batch_ligands, batch_labels

    def train(self):
        if (self.existing_model):
            model = joblib.load(self.model_file)
        else:
            model = SGDRegressor()
        
        chunk_index = 0
        lowest_err = float('inf')
        stop_iter = 0

        while True:
            self.shuffle_train_data()
            p_next = Process(target=self.next_train_chunk, args=(chunk_index,))
            self.running_process = True
            p_next.start()
            for chunk in range(self.total_train_chunks-1):
                print("starting chunk #"+str(chunk))
                self.train_receiver = self.train_queue.get(True)
                self.running_process = self.master.recv()
                chunk_size = self.train_receiver[1].shape[0]
                print(self.train_receiver[1])
                for batch in tqdm(range(
                    self.train_steps),  desc = "Training Model " + str(self.id) + " - Epoch " + str(self.epochs+1)):
                    print(self.running_process, self.train_queue.qsize())
                    ligands, labels = self.next_train_batch(chunk_size)
                    model.partial_fit(ligands, labels)
                    if (not self.running_process and self.train_queue.qsize() < self.max_queue_size):
                        if (chunk_index + self.chunk_size > self.train_members):

                        print("terminating finished process")
                        p_next.terminate()
                        print("beginning new process")
                        p_next = Process(target=self.next_train_chunk, args=(self.train_db_index,))
                        self.running_process = True
                        p_next.start()

            print("reached validation")
            val_err = self.validate(model)

            self.epochs+=1

            if (val_err < lowest_err):
                lowest_err = val_err
                joblib.dump(model, self.model_file)
                stop_iter = 0
                self.optimal_epochs = self.epochs
            else:
                stop_iter+=1

            if (stop_iter > self.stop_threshold):
                print("Finished Training...\n")
                print("\nValidation Set Error:", lowest_err)
                return

    def validate(self, model):
        self.shuffle_val_data()
        total_mse = 0
        print("started val process")
        p_next = Process(target=self.next_train_chunk, args=())
        p_next.start()
        for chunk in range(self.total_val_chunks):
            self.val_receiver = self.val_queue.get(True)
            chunk_size = self.val_receiver[1].shape[0]
            for batch in range(self.val_steps):
                if (not self.running_process and self.val_queue.qsize() < self.max_queue_size):
                    p_next.terminate()
                    p_next = Process(target=self.next_train_chunk)
                    self.running_process = True
                    p_next.start()

                ligands, labels = self.next_val_batch(chunk_size)
                predictions = model.predict(ligands)
                mse = mean_squared_error(labels, predictions)
                total_mse += mse


        return total_mse/(self.chunk_size*self.total_val_chunks)
