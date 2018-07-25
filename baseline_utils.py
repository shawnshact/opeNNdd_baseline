import numpy as np
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def train(baseline):
    concurrent_processes = Pool(processes=self.max_num_processes)
    if (baseline.existing_model):
        model = joblib.load(baseline.model_file)
    else:
        model = SGDRegressor()

    lowest_err = float('inf')
    stop_iter = 0

    while True:
        baseline.shuffle_train_data()
        results=[concurrent_processes.apply(baseline.next_train_chunk, args=(index,)) for index in range(baseline.train_db_index,baseline.chunk_size*baseline.max_num_processes,baseline.chunk_size)]

        if baseline.train_db_index > baseline.train_members:
            baseline.train_db_index = 0
        else:
            baseline.train_db_index += baseline.chunk_size*baseline.max_num_processes

        for chunk in range(len(results)):
            print("starting chunk #"+str(chunk))
            baseline.train_receiver = results[chunk]
            chunk_size = baseline.train_receiver[1].shape[0]
            print(baseline.train_receiver[1])
            for batch in tqdm(range(baseline.train_steps),  desc = "Training Model " + str(baseline.id) + " - Epoch " + str(baseline.epochs+1)):
                #print(baseline.running_process, baseline.train_queue.qsize())
                ligands, labels = baseline.next_train_batch(chunk_size)
                model.partial_fit(ligands, labels)

        print("reached validation")
        val_err = baseline.validate(model)

        baseline.epochs+=1

        if (val_err < lowest_err):
            lowest_err = val_err
            joblib.dump(model, baseline.model_file)
            stop_iter = 0
            baseline.optimal_epochs = baseline.epochs
        else:
            stop_iter+=1

        if (stop_iter > baseline.stop_threshold):
            print("Finished Training...\n")
            print("\nValidation Set Error:", lowest_err)
            return

def validate(baseline, model):
    baseline.shuffle_val_data()
    total_mse = 0
    print("started val process")
    p_next = Process(target=baseline.next_train_chunk, args=())
    p_next.start()
    for chunk in range(baseline.total_val_chunks):
        baseline.val_receiver = baseline.val_queue.get(True)
        chunk_size = baseline.val_receiver[1].shape[0]
        for batch in range(baseline.val_steps):
            if (not baseline.running_process and baseline.val_queue.qsize() < baseline.max_queue_size):
                p_next.terminate()
                p_next = Process(target=baseline.next_train_chunk)
                baseline.running_process = True
                p_next.start()

            ligands, labels = baseline.next_val_batch(chunk_size)
            predictions = model.predict(ligands)
            mse = mean_squared_error(labels, predictions)
            total_mse += mse


    return total_mse/(baseline.chunk_size*baseline.total_val_chunks)
