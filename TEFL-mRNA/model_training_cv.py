import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
import numpy as np
import pandas as pd
from time import time
from IPython.display import Image
import re
import random
import pdb
import tqdm
import pickle

from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Layer 
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.python.keras.utils import np_utils, generic_utils
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, History 
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers

from sklearn.preprocessing import LabelEncoder
import sklearn 
from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import OrdinalEncoder, label_binarize
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.callbacks import LearningRateScheduler

sys.path.append("/dir/to/model")
import utils as utils
from Motif import Motif
from Data import Data
from Model import Model
from One_Hot_Encoder import One_Hot_Encoder
from Alphabet_Encoder import Alphabet_Encoder
from Grid_Search import Grid_Search

def get_seq_array(seq_fl, seq_utr5, seq_cds, seq_utr3, max_len = 301):
    seq = seq_fl.replace('T', 'U')
    alpha = 'ACGU'
    array_seq = np.zeros((max_len, 4))
    array_region = np.zeros((max_len, 1))
    for i, val in enumerate(seq):
        if val not in 'ACGU':
            array_seq[i] = np.array([0]*4)
            continue
        else:
            index = alpha.index(val)
            array_seq[i][index] = 1
        i = i + 1
    array_region[0:len(seq_utr5)] = 0
    array_region[len(seq_utr5):(len(seq_utr5)+len(seq_cds))] = 1
    array_region[(len(seq_utr5)+len(seq_cds)):(len(seq_utr5)+len(seq_cds)+len(seq_utr3))] = 2
    new_array = np.concatenate((array_seq,array_region),axis=1)
    return new_array

def get_seqs_array(seqs_FL, seqs_UTR5, seqs_CDS, seqs_UTR3):
    max_len = len(max(seqs_FL, key=len))
    seq_list = []
    for i in range(0,len(seqs_FL)):
        seq_fl = seqs_FL[i]
        seq_utr5 = seqs_UTR5[i]
        seq_cds = seqs_CDS[i]
        seq_utr3 = seqs_UTR3[i]
        
        seq_array = get_seq_array(seq_fl, seq_utr5, seq_cds, seq_utr3, max_len)
        seq_list.append(seq_array)
    data = dict()
    data["seq"] = np.array(seq_list)
    data["Y"] = labels
    return data

def split_training_validation(classes, validation_size = 0.2, shuffle = False):
    """split sampels based on balnace classes"""
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)
        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl) # in-place shuffle
        # module and residual
        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split
    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    return training_indice, training_label, validation_indice, validation_label

def get_cnn_network_alhphabet(input_length):
    print('configure cnn network')
    nbfilter = 128
    model = Sequential()
    model.add(Conv1D(nbfilter,
                 11,
                 padding='valid',
                 activation='relu',
                 strides=3, 
                 input_shape=(input_length, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.5013842109122856))
    model.add(Bidirectional(LSTM(2*nbfilter)))
    model.add(Dropout(0.48589953225289306))
    model.add(Dense(2*nbfilter, activation='relu')) # relu
    model.add(BatchNormalization())
    model.add(Dropout(0.13623715759423677))
    return model

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def decay_schedule(epoch, lr):
    # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
    if (epoch % 5 == 0) and (epoch != 0):
        lr = lr * 0.9
    return lr

def run_network(model, total_hid, training, testing, y, validation, val_y, structure = None):
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    print('Start model training')
    lr_scheduler = LearningRateScheduler(decay_schedule,verbose = 1)
    earlystopper = EarlyStopping(monitor='val_loss', patience=200, verbose=1)
    model_hist = model.fit(training, y, batch_size=256, epochs=2000,
                           verbose=1, validation_data=(validation, val_y), 
                           callbacks=[earlystopper,lr_scheduler])#, class_weight = class_weight)
    predictions = model.predict(testing)
    species = ''
    model.save(os.path.join(output_folder + sample + '.CNN_BiLSTM_model.Fold' + str(fold_no)  + '.pkl'))
    return predictions, model, model_hist

def calculate_auc(net, hid, train, test, true_y, train_y, rf = False, validation = None, val_y = None, structure = None):
    if rf:
        print('running oli')
        predict, model = run_svm_classifier(train, train_y, test)
    else:
        predict, model, hist = run_network(net, hid, train, test, train_y, validation, val_y, structure = structure)
    auc = roc_auc_score(true_y, predict, multi_class='ovr')
    print("Test AUC: ", auc)
    return auc, model, hist, predict

def plot_auc(y_test, preds):
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return(plt.show())

# set GPU number
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


data_dir = "/dir/to/input_data/"
sample = "FL_regions.example.TPE_opt"
output_folder = "/dir/to/output/" + sample + "/"
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

seq_TE = pd.read_csv(data_dir + "FL_regions.example.TE_class.tsv", sep = "\t")

# read files
merge = seq_TE
merge = merge.dropna()
seqs_FL = list(merge["Sequences_FL"])
seqs_UTR5 = list(merge["Sequences_UTR5"])
seqs_CDS = list(merge["Sequences_CDS"])
seqs_UTR3 = list(merge["Sequences_UTR3"])
labels = list(merge["Class"])

merge["FL_len"] = [len(str(i)) for i in merge["Sequences_FL"]]


# split data to training, validation, and testing sets
training_data = get_seqs_array(seqs_FL, seqs_UTR5, seqs_CDS, seqs_UTR3)
seq_data = training_data["seq"]
train_Y = training_data["Y"]

training_indice, training_label, validation_indice, validation_label = split_training_validation(train_Y, validation_size = 0.1)
input_length = seq_data.shape[1]
seq_train = seq_data[training_indice]
cnn_validation = seq_data[validation_indice]


print(seq_train.shape)
print(cnn_validation.shape)

# Define the K-fold Cross Validator
aucs = []
accs = []
f1s = []
precisions = []
recalls = []

num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for training_indice, test_indice in kfold.split(seq_train, training_label):
    print('------------------------------------------------------------------------')
    print('Training for fold-' + str(fold_no))
    save_model_name = "Fold" + str(fold_no)

    cnn_train = seq_train[training_indice]
    testing = seq_train[test_indice]
    cnn_train_label = training_label[training_indice]
    test_label = training_label[test_indice]

    seq_net =  get_cnn_network_alhphabet(input_length)

    y, encoder = preprocess_labels(cnn_train_label)
    val_y, encoder = preprocess_labels(validation_label, encoder = encoder) 
    
    true_y = test_label.copy()
    true_y_onehot = preprocess_labels(true_y, encoder = encoder)

    print('Start predicting') 
    seq_hid = input_length
    struct_hid = input_length
    seq_auc, seq_model, seq_hist, seq_predict = calculate_auc(seq_net, seq_hid + struct_hid, cnn_train, testing, true_y, y, 
                                         validation = cnn_validation,
                                         val_y = val_y)

    #seq_predict
    pred_y_num=np.argmax(seq_predict, axis=-1)
    true_y_num = []
    for i in true_y:
        if i == 'low':
            true_y_num.append(2)
        elif i == "inter":
            true_y_num.append(1)
        else:
            true_y_num.append(0)
    true_y_bi = label_binarize(true_y_num, classes=[0,1,2])

    pred_y_num=np.argmax(seq_predict, axis=-1)
    true_y_num = []
    for i in true_y:
        if i == 'low':
            true_y_num.append(2)
        elif i == "inter":
            true_y_num.append(1)
        else:
            true_y_num.append(0)
    true_y_bi = label_binarize(true_y_num, classes=[0,1,2])
    true_y_bi = label_binarize(true_y, classes=["high","inter","low"])
    seq_predict_num = np.rint(seq_predict).astype(int)

    acc_all = accuracy_score(true_y_bi, seq_predict_num)
    f1_all = f1_score(true_y_bi, seq_predict_num, average="macro")
    precision_all = precision_score(true_y_bi, seq_predict_num, average="macro")
    recall_all = recall_score(true_y_bi, seq_predict_num, average="macro")
    auc_all = roc_auc_score(true_y, seq_predict, multi_class='ovr')

    print(save_model_name + "-Test ACC: " + str(acc_all))
    print(save_model_name + "-Test F1: " + str(f1_all))
    print(save_model_name + "-Test Precision: " + str(precision_all))
    print(save_model_name + "-Test Recall: " + str(recall_all))
    print(save_model_name + "-Test AUC: " + str(auc_all))

    aucs.append(auc_all)
    accs.append(acc_all)
    f1s.append(f1_all)
    precisions.append(precision_all)
    recalls.append(recall_all)

    train_loss = seq_hist.history['loss']
    val_loss   = seq_hist.history['val_loss']
    train_acc = seq_hist.history['accuracy']
    val_acc   = seq_hist.history['val_accuracy']
    xc = range(len(val_loss))
        
    df_train = pd.DataFrame(list(zip(xc, train_loss, val_loss, train_acc, val_acc)),
                                columns =['Epoch', 'train_loss', "val_loss", "train_acc", "val_acc"])
    df_pred = pd.DataFrame(list(zip(true_y, true_y_num, true_y_bi,true_y_bi[:,0],true_y_bi[:,1],true_y_bi[:,2], seq_predict,seq_predict[:,0],seq_predict[:,1],seq_predict[:,2], pred_y_num)),
                               columns =['true_y', 'true_y_num', "true_y_bi", "true_y_bi_0", "true_y_bi_1", "true_y_bi_2", "pred","pred_0","pred_1","pred_2","pred_y_num"])

    df_train.to_csv(output_folder + "Model_Training.fold_" + str(fold_no) + ".tsv",sep="\t",index=False)
    df_pred.to_csv(output_folder + "Model_Prediction.fold_" + str(fold_no) + ".tsv",sep="\t",index=False)
    fold_no = fold_no + 1

df_performance = pd.DataFrame(list(zip(aucs, accs, f1s, precisions, recalls)), columns =['AUC', 'ACC', 'F1', 'Precisions', 'Recall'])
df_performance.to_csv(output_folder + "Model_Performance."  + save_model_name + ".tsv", sep="\t",index=False)


