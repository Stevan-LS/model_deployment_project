import pickle
import numpy as np
import os

def read_cifar_batch(file_path):
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        data = batch[b'data'].astype(np.float32)
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        labels = np.array(batch[b'labels'], dtype=np.int64)
    return data, labels

def load_cifar_data(directory_path):
    data_list = []
    labels_list = []
    
    for batch_id in range(1, 6):
        file_path = os.path.join(directory_path, f'data_batch_{batch_id}')
        data, labels = read_cifar_batch(file_path)
        data_list.append(data)
        labels_list.append(labels)
    
    test_file_path = os.path.join(directory_path, 'test_batch')
    test_data, test_labels = read_cifar_batch(test_file_path)
    data_list.append(test_data)
    labels_list.append(test_labels)
    
    data = np.vstack(data_list)
    labels = np.hstack(labels_list)
    
    return data, labels