# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:04:10 2023

@author: m.ferrandezferrandez
"""

import os
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categories = 'auto')
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, SpatialDropout2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xlsxwriter
from sklearn.metrics import confusion_matrix, roc_auc_score

# For MLOps
import wandb
print("W&B: ", wandb.__version__)
from wandb.keras import WandbCallback



#%% THIS FIRST PART OF THE CODE EXTRACTS THE TRAINING AND DATA SCHEME THAT WAS USED FROM THE BEGINNING, SO THE IDs IN THE ORDER THEY BELONG TO EACH SUBSET

# we will load the patients ID for each subset (A to E) generated from the 2D CNN
path_excel = 'X:\\PETRA\\MARIA_AI\\Project_2\\python_scripts\\cnn\\training_excel_2DCNN\\'

#function to extract IDs and labels from excel files (per file)

def extract_IDs_labels(): #extraction of ID and label altogether
    train = []
    val = []
    for n in range(5):
        train_ids = []
        val_ids = []
        
        excel_file_name = 'CNN-nbr-outcomes-set'
        
        t_df = pd.read_excel(path_excel + excel_file_name + subset + '.xlsx', sheet_name = 'Train' + str(n))
        t_ids = list(t_df['Patient ID'])
        t_labels = list(t_df['True label'])
        
        for m in range(len(t_ids)):            
            train_ids.append([t_ids[m], t_labels[m]])
        
        
        v_df = pd.read_excel(path_excel + excel_file_name + subset + '.xlsx', sheet_name = 'Val' + str(n))
        v_ids = list(v_df['Patient ID'])
        v_labels = list(v_df['True label'])
        
        for l in range(len(v_ids)):
            val_ids.append([v_ids[l], v_labels[l]])
   
        train.append(train_ids)
        val.append(val_ids)
        
    return train, val


def extract_IDs(): #extraction of IDs
    train_ids = []
    val_ids = []
    for n in range(5):
        
        excel_file_name = 'CNN-nbr-outcomes-set'
        
        t_df = pd.read_excel(path_excel + excel_file_name + subset + '.xlsx', sheet_name = 'Train' + str(n))
        t_ids = list(t_df['Patient ID'])  
        train_ids.append(t_ids)
        
        
        v_df = pd.read_excel(path_excel + excel_file_name + subset + '.xlsx', sheet_name = 'Val' + str(n))
        v_ids = list(v_df['Patient ID'])
        val_ids.append(v_ids)
        
    return train_ids, val_ids


def extract_labels(): #extraction of labels and encoding
    train_labels = []
    val_labels = []
    for n in range(5):
        
        excel_file_name = 'CNN-nbr-outcomes-set'
        
        t_df = pd.read_excel(path_excel + excel_file_name + subset + '.xlsx', sheet_name = 'Train' + str(n))
        t_labels = list(t_df['True label'])  
        train_labels.append(t_labels)

        
        v_df = pd.read_excel(path_excel + excel_file_name + subset + '.xlsx', sheet_name = 'Val' + str(n))
        v_labels = list(v_df['True label'])
        val_labels.append(v_labels)

    return train_labels, val_labels





#array per training and val set, 5 arrays within each array
subsets = ['A', 'B', 'C', 'D', 'E']
subset = subsets[0]
train_A, val_A = extract_IDs()
trainlabs_A, valabs_A = extract_labels()
subset = subsets[1]
train_B, val_B = extract_IDs()
trainlabs_B, valabs_B = extract_labels()
subset = subsets[2]
train_C, val_C = extract_IDs()
trainlabs_C, valabs_C = extract_labels()
subset = subsets[3]
train_D, val_D = extract_IDs()
trainlabs_D, valabs_D = extract_labels()
subset = subsets[4]
train_E, val_E = extract_IDs()
trainlabs_E, valabs_E = extract_labels()


#function generating a whole subset out of the validation sets

def generate_subset(subset, labels):
    
    subset_final = []
    labels_final = []
    for n in range(5):
        subset_fold = subset[n]
        subset_labels = labels[n]
        for i in range(len(subset_fold)):
            subset_final.append(subset_fold[i])
            labels_final.append(subset_labels[i])
    
    return subset_final, labels_final

subset_A, labels_A = generate_subset(val_A, valabs_A)
subset_B, labels_B = generate_subset(val_B, valabs_B)
subset_C, labels_C = generate_subset(val_C, valabs_C)
subset_D, labels_D = generate_subset(val_D, valabs_D)
subset_E, labels_E = generate_subset(val_E, valabs_E)




#%% CONFIGURATIONS FOR WEIGHTS AND BIASES AND THE CNN

config_vois = dict(
    epochs = 200,
    learning_rate = 0.00005,
    lr_decay = 0.00001,
    loss_fn = 'categorical_crossentropy',
    metrics = ['categorical_accuracy'],
    dropout = 0.3,
    batch_size = 42
)



config = dict(
    epochs = 300,
    learning_rate = 0.00005,
    lr_decay = 0.00001,
    loss_fn = 'categorical_crossentropy',
    metrics = ['categorical_accuracy'],
    dropout = 0.3,
    batch_size = 42
)


#%% CNN DESIGN

def CNN(input_imgs, dropout): 

    classifier = Conv2D(16, (12,12), activation = 'relu', padding = 'valid', data_format = 'channels_last')(input_imgs) 
    classifier = SpatialDropout2D(dropout)(classifier)
    classifier = MaxPooling2D((3,3))(classifier)
    
    # Second convolution layer
    classifier = Conv2D(32, (9,9), activation = 'relu', padding = 'valid')(classifier)
    classifier = SpatialDropout2D(dropout)(classifier)
    classifier = MaxPooling2D((3,3))(classifier)
    
    # Third conv layer
    classifier = Conv2D(64, (6,6), activation = 'relu', padding = 'valid')(classifier)
    classifier = SpatialDropout2D(dropout)(classifier)
    classifier = MaxPooling2D((2,2))(classifier)
    
    # Fourth conv layer
    classifier = Conv2D(128, (3,3), activation = 'relu', padding = 'valid')(classifier)
    classifier = SpatialDropout2D(dropout)(classifier)

    # Flatten the layers
    classifier = GlobalAveragePooling2D()(classifier)

    return classifier



# Set up the classifier with the learning rate
def create_classifier(dropout):
    
    input_cor_imgs = Input(shape = (275, 200, 1)) # dtype = 'float64'
    input_sag_imgs = Input(shape = (275, 200, 1)) 

    cnn_coronal = CNN(input_cor_imgs, dropout)
    cnn_sagittal = CNN(input_sag_imgs, dropout)

    combined_input = concatenate([cnn_coronal, cnn_sagittal])
    classifier = Dense(2, activation = 'softmax')(combined_input) # 2 = dimensions of output
    classifier = Model(inputs = [input_cor_imgs, input_sag_imgs], outputs = classifier)


    # Add loss functions (bias) and optimizer (weight) (what values should the weight and the bias be updated)
    adam = Adam(lr = config['learning_rate'], decay = config['lr_decay'], beta_1 = 0.9, beta_2 = 0.999, epsilon = 10e-8) # decay = 0.0, amsgrad = True, Learning rate (step size), beta1 (exponential decay rate for the first moment estimates), beta2 (second), epsilon (small number to prevent division by zero), amsgrad (apply AMSGrad variant of this algorithm)
    classifier.compile(optimizer = adam, loss = config['loss_fn'], metrics = config['metrics'])
    
    return classifier


#%% DATA GENERATOR FUNCTION, THIS HELPS TO LOAD EACH OF THE FILES ONE BY ONE INTO THE FIT FUNCTION

# import our data generator class function
import sys
sys.path.append('X:\\PETRA\\MARIA_AI\\Project_2\\python_scripts\\cnn\\')
from DataGenerator_MIPs_cor_sag import DataGeneratorMIP, DataGeneratorLESION


#parameters for the classDataGenerator
params_train = {'dim': (275,200),
          'batch_size': 84,
          'n_classes': 2,
          'shuffle': False}

params_val = {'dim': (275,200),
          'batch_size': 21,
          'n_classes': 2,
          'shuffle': False}



#%% SOME FUNCTIONS FOR REPORTING RESULTS

#Create confusion Matrix and classification reports for training and validation
def sensitivity(TP, FN):
    sensitivity = TP / (TP + FN)
    sensitivity = round(sensitivity, 3)
    return sensitivity

def specificity(TN, FP):
    specificity = TN / (TN + FP)
    specificity = round(specificity, 3)
    return specificity


def classification_reports(workbook, job_type, classifier, gen_data, labels_di, save_path, IDs, count, best_epoch):
    predictions = classifier.predict(gen_data)
    print(len(labels_di))
    print(labels_di[0])
    print(len(predictions))
    print(predictions[0])
    pred_TTP1 = predictions[:,1]
    pred_TTP0 = predictions [:,0]
    conf_mat = confusion_matrix(labels_di, np.argmax(predictions, axis = 1))
    
    TN_cm = conf_mat[0][0]
    TP_cm = conf_mat[1][1]
    FP_cm = conf_mat[0][1]
    FN_cm = conf_mat[1][0]
    sens = sensitivity(TP_cm, FN_cm)
    spec = specificity(TN_cm, FP_cm)

    AUC = np.round((roc_auc_score(labels_di, pred_TTP1)), 3)

    worksheet = workbook.add_worksheet(str(job_type) + str(count))
    row = 1
    worksheet.write(0, 0, 'Patient ID')
    for ID in IDs:
        worksheet.write(row, 0, ID)
        row = row + 1
    row = 1
    worksheet.write(0, 1, 'True label')
    for lbl in labels_di:
        worksheet.write(row, 1, lbl)
        row = row + 1
    row = 1
    worksheet.write(0, 2, 'CNN P(TTP0)')
    for pred0 in pred_TTP0:
        worksheet.write(row, 2, np.round(pred0, 3))
        row = row + 1
    row = 1
    worksheet.write(0, 3, 'CNN P(TTP1)')
    for pred1 in pred_TTP1:
        worksheet.write(row, 3, np.round(pred1, 3))
        row = row + 1
    worksheet.write(0, 4, 'CNN AUC' + str(job_type))
    worksheet.write(1, 4, AUC)
    worksheet.write(0, 5, 'CNN sensitivity')
    worksheet.write(1, 5, sens)
    worksheet.write(0, 6, 'CNN specificity')
    worksheet.write(1, 6, spec)
    worksheet.write(3, 5, 'Best epoch')
    worksheet.write(4, 5, best_epoch)
    worksheet.write(5, 5, 'CNN confusion matrix:')
    worksheet.write(7, 5, 0)
    worksheet.write(8, 5, 1)
    worksheet.write(6, 6, 0)
    worksheet.write(6, 7, 1)
    worksheet.write(7, 6, TN_cm)
    worksheet.write(8, 6, FN_cm)
    worksheet.write(7, 7, FP_cm)
    worksheet.write(8, 7, TP_cm)



#%% TRAINING FUNCTION, WE USE MODELCHECKPOINT AND CALLBACKS TO KEEP AND SAVE BEST MODEL (DEPENDING ON THE LOSS OR ON THE ACCURACY), WE ALSO USE WANDA CALLBACK TO TRACK CURVES

def train_CNN_DG(config, training_set, validation_set, train_labels, val_labels, path, group, job_type, workbook_outcomes):
    
    save_path = '...'
    count = -1
    
    # Create worksheet with classifier outcomes 
    saved_path_workbook = save_path + workbook_outcomes
    workbook = xlsxwriter.Workbook(saved_path_workbook)

        
    for train, val, train_labs, val_labs in zip(training_set, validation_set, train_labels, val_labels):
        
        run = wandb.init(config=config, entity="...", project="...", group = group , job_type = job_type) #COMMENT THIS IF YOU ARE NOT USING W&B
            
        count = count + 1
        print('Iteration ' + str(count))
        print('\n')
        classifier = create_classifier(config['dropout'])
            
            
        IDs_train = train
            
        train_labs_arr = np.array(train_labs)
        train_labels_enc = train_labs_arr.reshape(np.size(train_labs_arr), 1)
        train_labels_enc = encoder.fit_transform(train_labels_enc).toarray().astype('float32')
        train_labels_di = train_labels_enc[:, 1].astype('int')
            
        IDs_val = val
            
        val_labs_arr = np.array(val_labs)
        val_labels_enc = val_labs_arr.reshape(np.size(val_labs_arr), 1)
        val_labels_enc = encoder.fit_transform(val_labels_enc).toarray().astype('float32')
        val_labels_di = val_labels_enc[:, 1].astype('int')
            
        
        train_gen_voi = DataGeneratorLESION(train, train_labels_enc, **params_train)
        val_gen_voi = DataGeneratorLESION(val, val_labels_enc, **params_val)
        
        train_gen = DataGeneratorMIP(train, train_labels_enc, **params_train)
        val_gen = DataGeneratorMIP(val, val_labels_enc, **params_val)
        
        
        #define callbacks 
        #callback for vois
        filepath_voi_loss = save_path + path + str(count) + '_voi.loss.best.h5'
        callback_loss_vois = ModelCheckpoint(filepath = filepath_voi_loss, save_weights_only = False, monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only = True)
        #filepath_voi_acc = save_path + path + str(count) + '_voi.acc.best.h5'
        #callback_acc_vois = ModelCheckpoint(filepath = filepath_voi_acc, save_weights_only = False, monitor = 'val_categorical_accuracy', mode = 'max', verbose = 1, save_best_only = True)
        
        #callback for scans (pretrained on vois)
        filepath_loss = save_path + path + str(count) + '.loss.best.h5'
        callback_loss = ModelCheckpoint(filepath = filepath_loss, save_weights_only = False, monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only = True)
        #filepath_acc = save_path + path + str(count) + '.acc.best.h5'
        #callback_acc = ModelCheckpoint(filepath = filepath_acc, save_weights_only = False, monitor = 'val_categorical_accuracy', mode = 'max', verbose = 1, save_best_only = True)
        
        
                    
        wandb_callback_loss_vois = WandbCallback(monitor='val_loss',
                               log_weights=True,
                               log_gradients = True,
                               training_data = train_gen_voi,
                               log_evaluation=True)
            
        
        wandb_callback_loss = WandbCallback(monitor='val_loss',
                               log_weights=True,
                               log_gradients = True,
                               training_data = train_gen,
                               log_evaluation=True)
        
            
        callbacks_vois = [callback_loss_vois, wandb_callback_loss_vois]    
        #callbacks_vois = [callback_loss_vois, callback_acc_vois, wandb_callback_loss_vois]
        callbacks = [callback_loss, wandb_callback_loss]
        #callbacks = [callback_loss, callback_acc, wandb_callback_loss]   
        
        print('Fitting CNN on vois')
        classifier.fit(train_gen_voi, epochs = config_vois['epochs'], validation_data = val_gen_voi, verbose = 1, callbacks = callbacks_vois)
        print('\n')
        
        run.finish() #COMMENT IF YOU ARE NOT USING W&B
            
        #load model trained on vois
        classifier = load_model(filepath_voi_loss)
        
        run = wandb.init(config=config, entity="...", project="...", group = group , job_type = job_type) #COMMENT IF YOU ARE NOT USING W&B
    
        #train vois model on scans
        print('Fitting pre-trained CNN on scans')
        model_final = classifier.fit(train_gen, epochs = config['epochs'], validation_data = val_gen, verbose = 1, callbacks = callbacks)
        print('\n')
        
        run.finish() #COMMENT IF NOT USING W&B
            
        loss = model_final.history['val_loss']
        best_epoch = np.argmin(loss) + 1
            
        model_final = load_model(filepath_loss)
        
        classification_reports(workbook = workbook, job_type = 'Training', classifier = model_final, gen_data = train_gen, 
                                   labels_di = train_labels_di, save_path = save_path, 
                                   IDs = IDs_train, count = count, best_epoch = best_epoch)
            
        classification_reports(workbook = workbook, job_type = 'Validation', classifier = model_final, gen_data = val_gen, 
                                   labels_di = val_labels_di, save_path = save_path, 
                                   IDs = IDs_val, count = count, best_epoch = best_epoch)
            
    workbook.close()  
                




# classifier_A = train_CNN(config, A_cor_train, A_cor_val, trainlabs_A, valabs_A, train_A, val_A, MIP_path = 'A_weights_MIP-CORnbr_3', group = 'subset A', job_type = 'train', workbook_outcomes = 'outcomes-subset-A.xlsx')
# classifier_B = train_CNN(config, B_cor_train, B_cor_val, trainlabs_B, valabs_B, train_B, val_B, MIP_path = 'B_weights_MIP-CORnbr_3', group = 'subset B', job_type = 'train', workbook_outcomes = 'outcomes-subset-B.xlsx')
train_CNN_DG(config, train_C, val_C, trainlabs_C, valabs_C, path = 'C_weights_CNN-OG', group = 'subset C', job_type = 'train', workbook_outcomes='outcomes-subset-C-CNN-OG.xlsx')
# classifier_D = train_CNN(config, D_cor_train, D_cor_val, trainlabs_D, valabs_D, train_D, val_D, MIP_path = 'D_weights_MIP-CORnbr_3', group = 'subset D', job_type = 'train', workbook_outcomes = 'outcomes-subset-D.xlsx')
# classifier_E = train_CNN(config, E_cor_train, E_cor_val, trainlabs_E, valabs_E, train_E, val_E, MIP_path = 'E_weights_MIP-CORnbr_3', group = 'subset E', job_type = 'train', workbook_outcomes = 'outcomes-subset-E.xlsx')





#%% FITTING FUNCTION

def fit_CNN_DG(subset, labels, path):
    
    save_path = '...'
    
    labs_arr = np.array(labels)
    labels_enc = labs_arr.reshape(np.size(labs_arr), 1)
    labels_enc = encoder.fit_transform(labels_enc).toarray().astype('float32')
        
    data_gen_voi = DataGeneratorLESION(subset, labels_enc, **params_train)
    data_gen = DataGeneratorMIP(subset, labels_enc, **params_train)
    
    classifier = create_classifier(config['dropout'])

    #define callbacks 
    filepath_voi = save_path + path + '_voi.FINAL.best.h5'
    callback_loss_voi = ModelCheckpoint(filepath = filepath_voi, save_weights_only = False, monitor = 'loss', mode = 'min', verbose = 1, save_best_only = True)
    #callback_acc_voi = ModelCheckpoint(filepath = filepath_voi, save_weights_only = False, monitor = 'categorical_accuracy', mode = 'max', verbose = 1, save_best_only = True)

    callback_list =  [callback_loss_voi]
    #callback_list =  [callback_loss_voi, callback_acc_voi]
    #train on vois
    classifier.fit(data_gen_voi, epochs = config_vois['epochs'], verbose = 1, callbacks = callback_list)
        
    #load model train on vois
    classifier = load_model(filepath_voi)

    filepath = save_path + path + 'loss.FINAL.best.h5'
    callback_loss = ModelCheckpoint(filepath = filepath, save_weights_only = False, monitor = 'loss', mode = 'min', verbose = 1, save_best_only = True)
    #callback_acc = ModelCheckpoint(filepath = filepath, save_weights_only = False, monitor = 'categorical_accuracy', mode = 'max', verbose = 1, save_best_only = True)
   
    #callback_list =  [callback_loss, callback_acc]
    callback_list =  [callback_loss]
    
    #pretrained model now training on scans
    classifier.fit(data_gen, epochs = config['epochs'], verbose = 1, callbacks = callback_list)
    
    classifier = load_model(filepath)
    
    return classifier


# classifier_A_final = fit_CNN(A_COR_nb, labels_A, path = 'A_weights_modelCORnbr_3')
# classifier_B_final = fit_CNN(B_COR_nb, labels_B, path = 'B_weights_modelCORnbr_3')
classifier_C_final = fit_CNN_DG(subset_C, labels_C, path = 'C_weights_CNN-OG')
# classifier_D_final = fit_CNN(D_COR_nb, labels_D, path = 'D_weights_modelCORnbr_3')
# classifier_E_final = fit_CNN(E_COR_nb, labels_E, path = 'E_weights_modelCORnbr_3')



# %% IF YOU PREFER TO NOT USE THE DATA GENERATOR AND LOAD THE DATA IN A SINGLE NUMPY ARRAY:

def array_of_MIPs(path, ID_list):
    list0 = ID_list[0]
    list1 = ID_list[1]
    list2 = ID_list[2]
    list3 = ID_list[3]
    list4 = ID_list[4]

    cor_MIP0 = []
    for patient_ID in list0:
        MIP = np.load(path + '\\' + str(patient_ID) + '_MIPnbr.npy')
        MIP = MIP.reshape(275, 200, 1)
        cor_MIP0.append(MIP)

    cor_MIP1 = []
    for patient_ID in list1:
        MIP = np.load(path + '\\' + str(patient_ID) + '_MIPnbr.npy')
        MIP = MIP.reshape(275, 200, 1)
        cor_MIP1.append(MIP)

    cor_MIP2 = []
    for patient_ID in list2:
        MIP = np.load(path + '\\' + str(patient_ID) + '_MIPnbr.npy')
        MIP = MIP.reshape(275, 200, 1)
        cor_MIP2.append(MIP)

    cor_MIP3 = []
    for patient_ID in list3:
        MIP = np.load(path + '\\' + str(patient_ID) + '_MIPnbr.npy')
        MIP = MIP.reshape(275, 200, 1)
        cor_MIP3.append(MIP)

    cor_MIP4 = []
    for patient_ID in list4:
        MIP = np.load(path + '\\' + str(patient_ID) + '_MIPnbr.npy')
        MIP = MIP.reshape(275, 200, 1)
        cor_MIP4.append(MIP)

    cormip0 = np.array(cor_MIP0)
    cormip1 = np.array(cor_MIP1)
    cormip2 = np.array(cor_MIP2)
    cormip3 = np.array(cor_MIP3)
    cormip4 = np.array(cor_MIP4)

    cormip = np.array([cormip0, cormip1, cormip2, cormip3, cormip4])

    return cormip


def array_of_lesion_MIPs(path, ID_list):
    list0 = ID_list[0]
    list1 = ID_list[1]
    list2 = ID_list[2]
    list3 = ID_list[3]
    list4 = ID_list[4]

    cor_MIP0 = []
    for patient_ID in list0:
        MIP = np.load(path + '\\' + str(patient_ID) + '_VOI_MIP.npy')
        MIP = MIP.reshape(275, 200, 1)
        cor_MIP0.append(MIP)

    cor_MIP1 = []
    for patient_ID in list1:
        MIP = np.load(path + '\\' + str(patient_ID) + '_VOI_MIP.npy')
        MIP = MIP.reshape(275, 200, 1)
        cor_MIP1.append(MIP)

    cor_MIP2 = []
    for patient_ID in list2:
        MIP = np.load(path + '\\' + str(patient_ID) + '_VOI_MIP.npy')
        MIP = MIP.reshape(275, 200, 1)
        cor_MIP2.append(MIP)

    cor_MIP3 = []
    for patient_ID in list3:
        MIP = np.load(path + '\\' + str(patient_ID) + '_VOI_MIP.npy')
        MIP = MIP.reshape(275, 200, 1)
        cor_MIP3.append(MIP)

    cor_MIP4 = []
    for patient_ID in list4:
        MIP = np.load(path + '\\' + str(patient_ID) + '_VOI_MIP.npy')
        MIP = MIP.reshape(275, 200, 1)
        cor_MIP4.append(MIP)

    cormip0 = np.array(cor_MIP0)
    cormip1 = np.array(cor_MIP1)
    cormip2 = np.array(cor_MIP2)
    cormip3 = np.array(cor_MIP3)
    cormip4 = np.array(cor_MIP4)

    cormip = np.array([cormip0, cormip1, cormip2, cormip3, cormip4])

    return cormip



path_cor = 'X:\\PETRA\\MARIA_AI\\Project_2\\data\\HOVON84\\MIPs_nbr\\cor'
path_sag = 'X:\\PETRA\\MARIA_AI\\Project_2\\data\\HOVON84\\MIPs_nbr\\sag'
cormipA = array_of_MIPs(path_cor, train_A)
sagmipA = array_of_MIPs(path_sag, train_A)
cormipB = array_of_MIPs(path_cor, train_B)
sagmipB = array_of_MIPs(path_sag, train_B)
cormipC = array_of_MIPs(path_cor, train_C)
sagmipC = array_of_MIPs(path_sag, train_C)
cormipD = array_of_MIPs(path_cor, train_D)
sagmipD = array_of_MIPs(path_sag, train_D)
cormipE = array_of_MIPs(path_cor, train_E)
sagmipE = array_of_MIPs(path_sag, train_E)


cormipA_val = array_of_MIPs(path_cor, val_A)
sagmipA_val = array_of_MIPs(path_sag, val_A)
cormipB_val = array_of_MIPs(path_cor, val_B)
sagmipB_val = array_of_MIPs(path_sag, val_B)
cormipC_val = array_of_MIPs(path_cor, val_C)
sagmipC_val = array_of_MIPs(path_sag, val_C)
cormipD_val = array_of_MIPs(path_cor, val_D)
sagmipD_val = array_of_MIPs(path_sag, val_D)
cormipE_val = array_of_MIPs(path_cor, val_E)
sagmipE_val = array_of_MIPs(path_sag, val_E)

path_cor_lesion = 'X:\\PETRA\\MARIA_AI\\Project_2\\data\\HOVON84\\vois_mip\\cor'
path_sag_lesion = 'X:\\PETRA\\MARIA_AI\\Project_2\\data\\HOVON84\\vois_mip\\sag'

cormip_lesionA = array_of_lesion_MIPs(path_cor_lesion, train_A)
sagmip_lesionA = array_of_lesion_MIPs(path_sag_lesion, train_A)
cormip_lesionB = array_of_lesion_MIPs(path_cor_lesion, train_B)
sagmip_lesionB = array_of_lesion_MIPs(path_sag_lesion, train_B)
cormip_lesionC = array_of_lesion_MIPs(path_corlesion, train_C)
sagmip_lesionC = array_of_lesion_MIPs(path_sag_lesion, train_C)
cormip_lesionD = array_of_lesion_MIPs(path_cor_lesion, train_D)
sagmip_lesionD = array_of_lesion_MIPs(path_sag_lesion, train_D)
cormip_lesionE = array_of_lesion_MIPs(path_cor_lesion, train_E)
sagmip_lesionE = array_of_lesion_MIPs(path_sag_lesion, train_E)

cormip_lesionA_val = array_of_lesion_MIPs(path_cor_lesion, val_A)
sagmip_lesionA_val = array_of_lesion_MIPs(path_sag_lesion, val_A)
cormip_lesionB_val = array_of_lesion_MIPs(path_cor_lesion, val_B)
sagmip_lesionB_val = array_of_lesion_MIPs(path_sag_lesion, val_B)
cormip_lesionC_val = array_of_lesion_MIPs(path_cor_lesion, val_C)
sagmip_lesionC_val = array_of_lesion_MIPs(path_sag_lesion, val_C)
cormip_lesionD_val = array_of_lesion_MIPs(path_cor_lesion, val_D)
sagmip_lesionD_val = array_of_lesion_MIPs(path_sag_lesion, val_D)
cormip_lesionE_val = array_of_lesion_MIPs(path_cor_lesion, val_E)
sagmip_lesionE_val = array_of_lesion_MIPs(path_sag_lesion, val_E)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True,
                         figsize=(8, 4))

axes[0].imshow(1 - cormipA[0][0], cmap=plt.cm.gray)
axes[0].set_title('coronal view, slice 100')

axes[1].imshow(1 - sagmipA[0][0], cmap=plt.cm.gray)
axes[1].set_title('generated MIP')

axes[2].imshow(1 - cormip_lesionA[0][0], cmap=plt.cm.gray)
axes[2].set_title('generated MIP')

plt.tight_layout()
plt.show()

for root, patients, files in os.walk(path_cor):
    cor_MIP0 = []
    # print(patient_ID)
    for patient_ID in train_A[0]:
        print(patient_ID)
        MIP = np.load(path_cor + '\\' + str(patient_ID) + '_MIPnbr.npy')
        MIP = MIP.reshape(275, 200, 1)
        cor_MIP0.append(MIP)

cor_MIParr = np.array(cor_MIP0)

def train_CNN(config, training_set_cor, validation_set_cor, training_set_sag, validation_set_sag,
                training_set_corlesion, validation_set_corlesion, training_set_saglesion, validation_set_saglesion,
                train_labels, val_labels, IDs_train, IDs_val, path, group, job_type, workbook_outcomes):

    save_path = '...'
    count = -1

    # Create worksheet with classifier outcomes
    saved_path_workbook = save_path + workbook_outcomes
    workbook = xlsxwriter.Workbook(saved_path_workbook)

    for train_cor, val_cor, train_sag, cor_sag, train_corlesion, val_corlesion, train_saglesion, cor_saglesion, train_ids, val_ids, train_labs, val_labs in zip(training_set_cor, validation_set_cor,
                                                                                                                                                                training_set_sag, validation_set_sag,
                                                                                                                                                                training_set_corlesion, validation_set_corlesion,
                                                                                                                                                                training_set_saglesion, validation_set_saglesion,

                                                                                                                                                               IDs_train, IDs_val, train_labels, val_labels):
        run = wandb.init(config=config, entity="...", project="...", group=group, job_type=job_type)  # COMMENT THIS IF YOU ARE NOT USING W&B

        count = count + 1
        print('Iteration ' + str(count))
        print('\n')
        classifier = create_classifier(config['dropout'])

        corlesion_train = train_corlesion
        saglesion_train = train_saglesion
        cormips_train = train_cor
        sagmips_train = train_sag
        train_ids = IDs_train


        train_labs_arr = np.array(train_labs)
        train_labels_enc = train_labs_arr.reshape(np.size(train_labs_arr), 1)
        train_labels_enc = encoder.fit_transform(train_labels_enc).toarray().astype('float32')
        train_labels_di = train_labels_enc[:, 1].astype('int')

        corlesion_val = val_corlesion
        saglesion_val = val_saglesion
        cormips_val = val_cor
        sagmips_val = val_sag
        val_ids = IDs_val

        val_labs_arr = np.array(val_labs)
        val_labels_enc = val_labs_arr.reshape(np.size(val_labs_arr), 1)
        val_labels_enc = encoder.fit_transform(val_labels_enc).toarray().astype('float32')
        val_labels_di = val_labels_enc[:, 1].astype('int')



        # define callbacks
        # callback for vois
        filepath_voi_loss = save_path + path + str(count) + '_voi.loss.best.h5'
        callback_loss_vois = ModelCheckpoint(filepath=filepath_voi_loss, save_weights_only=False, monitor='val_loss',
                                             mode='min', verbose=1, save_best_only=True)
        # filepath_voi_acc = save_path + path + str(count) + '_voi.acc.best.h5'
        # callback_acc_vois = ModelCheckpoint(filepath = filepath_voi_acc, save_weights_only = False, monitor = 'val_categorical_accuracy', mode = 'max', verbose = 1, save_best_only = True)

        # callback for scans (pretrained on vois)
        filepath_loss = save_path + path + str(count) + '.loss.best.h5'
        callback_loss = ModelCheckpoint(filepath=filepath_loss, save_weights_only=False, monitor='val_loss', mode='min',
                                        verbose=1, save_best_only=True)
        # filepath_acc = save_path + path + str(count) + '.acc.best.h5'
        # callback_acc = ModelCheckpoint(filepath = filepath_acc, save_weights_only = False, monitor = 'val_categorical_accuracy', mode = 'max', verbose = 1, save_best_only = True)

        wandb_callback_loss_vois = WandbCallback(monitor='val_loss',
                                                 log_weights=True,
                                                 log_gradients=True,
                                                 training_data=train_gen_voi,
                                                 log_evaluation=True)

        wandb_callback_loss = WandbCallback(monitor='val_loss',
                                            log_weights=True,
                                            log_gradients=True,
                                            training_data=train_gen,
                                            log_evaluation=True)

        callbacks_vois = [callback_loss_vois, wandb_callback_loss_vois]
        # callbacks_vois = [callback_loss_vois, callback_acc_vois, wandb_callback_loss_vois]
        callbacks = [callback_loss, wandb_callback_loss]
        # callbacks = [callback_loss, callback_acc, wandb_callback_loss]

        print('Fitting CNN on vois')
        classifier.fit([corlesion_train, saglesion_train], train_labels_enc, epochs=config_vois['epochs'],
                       validation_data= ([corlesion_val, saglesion_val], val_labels_enc), verbose=1, callbacks=callbacks_vois)
        print('\n')

        run.finish()  # COMMENT IF YOU ARE NOT USING W&B

        # load model trained on vois
        classifier = load_model(filepath_voi_loss)

        run = wandb.init(config=config, entity="...", project="...", group=group, job_type=job_type)  # COMMENT IF YOU ARE NOT USING W&B

        # train vois model on scans
        print('Fitting pre-trained CNN on scans')
        classifier.fit([cormips_train, sagmips_train], train_labels_enc, epochs=config_vois['epochs'],
                       validation_data= ([cormips_val, sagmips_val], val_labels_enc), verbose=1, callbacks=callbacks_vois)
        print('\n')

        run.finish()  # COMMENT IF NOT USING W&B

        loss = model_final.history['val_loss']
        best_epoch = np.argmin(loss) + 1

        model_final = load_model(filepath_loss)

        classification_reports(workbook=workbook, job_type='Training', classifier=model_final, gen_data=train_gen,
                               labels_di=train_labels_di, save_path=save_path,
                               IDs=IDs_train, count=count, best_epoch=best_epoch)

        classification_reports(workbook=workbook, job_type='Validation', classifier=model_final, gen_data=val_gen,
                               labels_di=val_labels_di, save_path=save_path,
                               IDs=IDs_val, count=count, best_epoch=best_epoch)

    workbook.close()


# classifier_A = train_CNN(config, A_cor_train, A_cor_val, trainlabs_A, valabs_A, train_A, val_A, MIP_path = 'A_weights_MIP-CORnbr_3', group = 'subset A', job_type = 'train', workbook_outcomes = 'outcomes-subset-A.xlsx')
# classifier_B = train_CNN(config, B_cor_train, B_cor_val, trainlabs_B, valabs_B, train_B, val_B, MIP_path = 'B_weights_MIP-CORnbr_3', group = 'subset B', job_type = 'train', workbook_outcomes = 'outcomes-subset-B.xlsx')
train_CNN(config,
          cormipC, cormipC_val, sagmip_C, sagmipC_val, #mips
          cormip_lesionC, cormip_lesionC_val, sagmip_lesionC, sagmip_lesionC_val, #lesions
          trainlabs_C, valabs_C, train_C, val_C, #labels and ids
          path='C_weights_CNN-OG', group='subset C', job_type='train', workbook_outcomes='outcomes-subset-C-CNN-OG-noDG.xlsx')
# classifier_D = train_CNN(config, D_cor_train, D_cor_val, trainlabs_D, valabs_D, train_D, val_D, MIP_path = 'D_weights_MIP-CORnbr_3', group = 'subset D', job_type = 'train', workbook_outcomes = 'outcomes-subset-D.xlsx')
# classifier_E = train_CNN(config, E_cor_train, E_cor_val, trainlabs_E, valabs_E, train_E, val_E, MIP_path = 'E_weights_MIP-CORnbr_3', group = 'subset E', job_type = 'train', workbook_outcomes = 'outcomes-subset-E.xlsx')





# %% FITTING FUNCTION

def fit_CNN(subset, labels, path):
    save_path = '...'

    labs_arr = np.array(labels)
    labels_enc = labs_arr.reshape(np.size(labs_arr), 1)
    labels_enc = encoder.fit_transform(labels_enc).toarray().astype('float32')

    classifier = create_classifier(config['dropout'])

    # define callbacks
    filepath_voi = save_path + path + '_voi.FINAL.best.h5'
    callback_loss_voi = ModelCheckpoint(filepath=filepath_voi, save_weights_only=False, monitor='loss', mode='min',
                                        verbose=1, save_best_only=True)
    # callback_acc_voi = ModelCheckpoint(filepath = filepath_voi, save_weights_only = False, monitor = 'categorical_accuracy', mode = 'max', verbose = 1, save_best_only = True)

    callback_list = [callback_loss_voi]
    # callback_list =  [callback_loss_voi, callback_acc_voi]
    # train on vois
    classifier.fit(data_gen_voi, epochs=config_vois['epochs'], verbose=1, callbacks=callback_list)

    # load model train on vois
    classifier = load_model(filepath_voi)

    filepath = save_path + path + 'loss.FINAL.best.h5'
    callback_loss = ModelCheckpoint(filepath=filepath, save_weights_only=False, monitor='loss', mode='min', verbose=1,
                                    save_best_only=True)
    # callback_acc = ModelCheckpoint(filepath = filepath, save_weights_only = False, monitor = 'categorical_accuracy', mode = 'max', verbose = 1, save_best_only = True)

    # callback_list =  [callback_loss, callback_acc]
    callback_list = [callback_loss]

    # pretrained model now training on scans
    classifier.fit(data_gen, epochs=config['epochs'], verbose=1, callbacks=callback_list)

    classifier = load_model(filepath)

    return classifier


# classifier_A_final = fit_CNN(A_COR_nb, labels_A, path = 'A_weights_modelCORnbr_3')
# classifier_B_final = fit_CNN(B_COR_nb, labels_B, path = 'B_weights_modelCORnbr_3')
#classifier_C_final = fit_CNN(subset_C, labels_C, path='C_weights_CNN-OG')
# classifier_D_final = fit_CNN(D_COR_nb, labels_D, path = 'D_weights_modelCORnbr_3')
# classifier_E_final = fit_CNN(E_COR_nb, labels_E, path = 'E_weights_modelCORnbr_3')
