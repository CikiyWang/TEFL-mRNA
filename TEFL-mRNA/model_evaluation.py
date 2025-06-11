import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import sys
import numpy as np
import pandas as pd
from time import time
from IPython.display import Image
import re
import random
import pdb

from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Layer
from keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.utils import np_utils, generic_utils
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import OrdinalEncoder, label_binarize
import sklearn.metrics as metrics
import tensorflow as tf
import keras

sys.path.append("/dir/to/model")
import utils as utils
from Motif import Motif
from Data import Data
from Model import Model
from One_Hot_Encoder import One_Hot_Encoder
from Alphabet_Encoder import Alphabet_Encoder
from Grid_Search import Grid_Search

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

def get_seq_array(seq_fl, seq_utr5, seq_cds, seq_utr3, max_len):
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
        if i >= max_len:
            break
    array_region[0:len(seq_utr5)] = 0
    array_region[len(seq_utr5):(len(seq_utr5)+len(seq_cds))] = 1
    array_region[(len(seq_utr5)+len(seq_cds)):(len(seq_utr5)+len(seq_cds)+len(seq_utr3))] = 2
    new_array = np.concatenate((array_seq,array_region),axis=1)
    return new_array

def get_seqs_array(seqs_FL, seqs_UTR5, seqs_CDS, seqs_UTR3, max_len):
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
    return data

from tensorflow import keras

############################
# load saved model

model = keras.models.load_model("/dir/to/save_model_tpe_opt")


############################
# modify the input sequences file names and output names
############################

file_in = "SNVs_FL_example.tsv"
file_out = "pred_SNVs_FL_example.tsv"
merge_sub = pd.read_csv(file_in, sep = "\t")
merge_sub = merge_sub.dropna()
max_lens = 19940

testing_data = get_seqs_array(list(merge_sub["Sequences_FL"]),
    list(merge_sub["Sequences_UTR5"]),
    list(merge_sub["Sequences_CDS"]),
    list(merge_sub["Sequences_UTR3"]), max_lens)
predictions = model.predict(testing_data["seq"])
df_sub = pd.DataFrame(predictions, columns=['pred_0', 'pred_1','pred_2'])
result = pd.concat([merge_sub, df_sub.set_index(merge_sub.index)], axis=1).reset_index(drop=True)
result.to_csv(file_out,sep = "\t")
