#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, scipy as sp, scipy.stats, os, sys
import tensorflow as tf
import glob
import math
from scipy import signal

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding data files...')


    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)
    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(model_folder+'/data', exist_ok=True)
    os.makedirs(model_folder+'/batch_data', exist_ok=True)

    classes = ['Present', 'Unknown', 'Absent']

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    fs = 4000
    frame_length = 5
    batch_size = 25
    export_data(patient_files, classes, frame_length, fs, data_folder, model_folder)
    create_batch_directory(model_folder+'/data', model_folder+'/batch_data', batch_size)
    
    seperator = os.sep
    samples = glob.glob(model_folder+'/batch_data/*.npy')
    samples.sort(key=lambda x: int(x.split(seperator)[-1].split(".")[0]))
    model = stateful_model(input_shape = (1*batch_size, fs*frame_length, 1))    
    stateful_generator = StatefulDataGenerator(samples,batch_size=batch_size, seperator=seperator, Tx=fs*frame_length, frame_length=frame_length)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    METRICS = ['accuracy']
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=METRICS)#,sample_weight_mode="temporal")
    history = model.fit(stateful_generator, 
                    epochs=10, 
                    max_queue_size=10,            
                    workers=1,                        
                    use_multiprocessing=False,       
                    shuffle=False,
                    initial_epoch=0)
    model.save(model_folder+'/my_model.h5')

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    model = tf.keras.models.load_model(model_folder+'/my_model.h5')
    model_predict = stateful_model(input_shape = (1, 20000, 1))
    model_predict.set_weights(model.get_weights())
    return model_predict

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_model(model, data, recordings, verbose):
    fs = 4000
    frame_length = 5
    try:
        fs = int(data.split('\n')[0].split(' ')[2])
    except:
        pass 
    print(data.split('\n')[0].split()[0]) 
    probabilities = np.zeros(3) 
    present_probabilities = None    
    for recording in recordings:
        #model.reset_states()
        sig_pad = recording
        sig_len = len(sig_pad)
        if sig_len < fs*frame_length :
            for xx in range(fs*frame_length - sig_len):
                sig_pad.append(0)
        pad_remainder = fs*frame_length-int(sig_len%(fs*frame_length))
        if pad_remainder < fs*frame_length:
            sig_pad = np.concatenate([sig_pad, sig_pad[0:pad_remainder]])
        maxData = max(sig_pad)
        samples_len = int(len(sig_pad)/(fs*frame_length))
        
        X = np.zeros([samples_len, fs*frame_length]) 
        for k in range(samples_len):    
            dataSeg = sig_pad[k*(fs*frame_length):(k+1)*(fs*frame_length)]
            dataSeg = np.multiply(dataSeg, 1/maxData)
            dataSeg = signal.resample(dataSeg, 4000*frame_length)
            X[k] = dataSeg
        
        normData = X[0]
        for i in range(1,X.shape[0]):
            normData = np.concatenate((normData, X[i]))
        mean = np.mean(normData)  
        std = np.std(normData) 
        if std == 0.0:
            std = 1 
        X = (X-mean)/std        
        for k in range(samples_len):    
             X_pred = np.zeros([1, 4000*frame_length, 1])
             X_pred[0,] = np.expand_dims(X[k], axis=1)
             y_pred = model.predict(X_pred,verbose=0)                    

        classes=['Present','Unknown','Absent']
        probabilities = probabilities + y_pred[0]
        if y_pred[0][0] > 0.5:
            present_probabilities = y_pred[0]
            print(present_probabilities)

    probabilities = probabilities/len(recordings) 
    if present_probabilities is not None:
        probabilities = present_probabilities     
    # Choose label with higher probability.
    labels = np.zeros(len(classes), dtype=np.int_)
    idx = np.argmax(probabilities)
    labels[idx] = 1

    return classes, labels, probabilities

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

def export_data(patient_files, classes, frame_length, fs, data_folder, model_folder):
    for i in range(len(patient_files)):
        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        patient = current_patient_data.split()[0]
        label = get_label(current_patient_data)
        # pad recordings and split them in 5s segments 
        current_recordings, current_frequencies = load_recordings(data_folder, current_patient_data, get_frequencies=True)
        for j in range(len(current_recordings)):
            os.makedirs(model_folder+'/data/'+patient+'X'+str(j), exist_ok=True)
            sig_pad = current_recordings[j] 
            sig_len = len(sig_pad)
            if sig_len < current_frequencies[j]*frame_length :
                for xx in range(current_frequencies[j]*frame_length - sig_len):
                    sig_pad.append(0)
            pad_remainder = current_frequencies[j]*frame_length-int(sig_len%(current_frequencies[j]*frame_length))
            if pad_remainder < current_frequencies[j]*frame_length:
                sig_pad = np.concatenate([sig_pad, sig_pad[0:pad_remainder]])
            maxData = max(sig_pad)
            X = np.zeros([int(len(sig_pad)/(current_frequencies[j]*frame_length)), fs*frame_length]) 
            for k in range(int(len(sig_pad)/(current_frequencies[j]*frame_length))):    
                dataSeg = sig_pad[k*(current_frequencies[j]*frame_length):(k+1)*(current_frequencies[j]*frame_length)]
                dataSeg = np.multiply(dataSeg, 1/maxData)
                dataSeg = signal.resample(dataSeg, fs*frame_length)
                X[k] = dataSeg

            normData = X[0]
            for i in range(1,X.shape[0]):
                normData = np.concatenate((normData, X[i]))
            mean = np.mean(normData)  
            std = np.std(normData) 
            if std == 0.0:
                std = 1 
            X = (X-mean)/std
            for k in range(int(len(sig_pad)/(current_frequencies[j]*frame_length))):             
                np.save(model_folder+'/data/'+patient+'X'+str(j)+'/'+patient+'X'+str(j)+'_'+str(k*frame_length)+'_'+str((k+1)*frame_length)+'_'+str(label)+".npy", X[k]) 
                
def create_batch_directory(dataDir, batchDir, batch_size):
    seperator = os.sep
    sample_dirs = [el.split(seperator)[-1]for el in glob.glob(dataDir+'/*')]
    sample_dirs.sort(key=lambda x: len(glob.glob(dataDir+'/'+x+'/*')))
    reverse = False

    sample_dir_map = {}
    for dir in sample_dirs :
        sample_dir_map[dir] = [el.split(seperator)[-1 ]for el in glob.glob(dataDir+'/'+dir+'/*.npy')]
        sample_dir_map[dir].sort(key=lambda x: int(float(x.split('_')[1])), reverse=reverse)

    batches = math.ceil(len(sample_dirs)/batch_size)
    sample_dirs.extend(sample_dirs)
    counter = 0
    for i in range(batches) :
        dirs = sample_dirs[batch_size*i:batch_size*(i+1)]
        max_file_num = max([len(sample_dir_map[el]) for el in dirs])
        for j in range(max_file_num) :
            f = open(batchDir+'/'+str(counter)+'.txt','w')
            data = np.load(dataDir+'/'+dirs[0]+'/'+sample_dir_map[dirs[0]][j%len(sample_dir_map[dirs[0]])])
            f.write(sample_dir_map[dirs[0]][j%len(sample_dir_map[dirs[0]])].split('.npy')[0]+'\n') 
            for dir in dirs[1:] :
                tmp = np.load(dataDir+'/'+dir+'/'+sample_dir_map[dir][j%len(sample_dir_map[dir])])
                data = np.concatenate((data,tmp)) 
                f.write(sample_dir_map[dir][j%len(sample_dir_map[dir])].split('.npy')[0]+'\n')         
            f.close()
            data32 = data.astype('float32')
            np.save(batchDir+'/'+str(counter),data32)  
            counter = counter + 1 

class StatefulDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_files, classes=['Present','Unknown','Absent'], batch_size=32, seperator = "/", Tx=20000, frame_length=5):
        'Initialization'
        self.classes = classes
        self.batch_size = batch_size
        self.Tx = Tx
        self.frame_length = frame_length
        self.seperator = seperator
        self.list_files = list_files
        print("Found "+str(len(list_files))+" samples")
        self.sequences = [*range(self.batch_size)]
        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.list_files)))
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index:(index+1)]
        # Find list of IDs
        list_files_temp = [self.list_files[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_files_temp)
        return X, y
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_files))
    def __data_generation(self, list_files_temp):
        # Initialization
        X = np.zeros([1*self.batch_size, self.Tx, 1])
        y = np.zeros([1*self.batch_size, 3, 1])
        # Generate data
        ID = list_files_temp[0]
        data = np.load(ID)
        length = int(data.shape[0]/self.batch_size)
        f = open(ID.split('.npy')[0]+'.txt')
        event_files = f.readlines()
        f.close()
        for i, ID in enumerate(self.sequences):
            # Store sample
            X[i,] = np.expand_dims(data[ID*length:(ID+1)*length], axis=1)
            # Store target
            label = event_files[ID].split("\n")[0].split("_")[-1] 
            current_labels = np.zeros((len(self.classes),1), dtype=int)
            if label in self.classes:
                j = self.classes.index(label)
                current_labels[j] = 1
            y[i] = current_labels
        return X, y   
    
def stateful_model(input_shape):
     X_input = tf.keras.Input(batch_shape = input_shape)
     X = tf.keras.layers.BatchNormalization()(X_input)
     X = tf.keras.layers.Conv1D(filters=128,kernel_size=5,strides=2)(X)                                 
     X = tf.keras.layers.BatchNormalization()(X)                                 
     X = tf.keras.layers.Activation("relu")(X)                                 
     X = tf.keras.layers.Dropout(rate=0.2,noise_shape=(1, 1, 128))(X)
     X = tf.keras.layers.Conv1D(filters=128,kernel_size=5,strides=2)(X)                                 
     X = tf.keras.layers.BatchNormalization()(X)                                 
     X = tf.keras.layers.Activation("relu")(X)                                
     X = tf.keras.layers.Dropout(rate=0.2,noise_shape=(1, 1, 128))(X)
     X = tf.keras.layers.Conv1D(filters=128,kernel_size=5,strides=2)(X)                                 
     X = tf.keras.layers.BatchNormalization()(X)                                 
     X = tf.keras.layers.Activation("relu")(X)                                
     X = tf.keras.layers.Dropout(rate=0.2,noise_shape=(1, 1, 128))(X)   
     X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=128, stateful=True),merge_mode='ave')(X)    
     X = tf.keras.layers.Dense(3, activation = "softmax")(X) 
     model = tf.keras.Model(inputs = X_input, outputs = X)
     model.summary()
     return model