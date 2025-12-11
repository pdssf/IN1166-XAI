# -*- coding: utf-8 -*-
"""

Steps performed in code:    

    - Read in train and test sets (Most of this done in the utility file)
    - Train XGBoost model to perform supervised intrusion detection
    - Use TreeSHAP to explain model predictions (across training set and test set)
    - Train autoencoder module using explanations from training set
    - Perform anomaly detection on explanations from test set based on reconstruction error of the autoencoder


"""

import os
os.chdir('./')              	# location where files are stored (main + utility)
save_loc = './results_car_hacking_model'    			# location to save results
data_loc = './results_car_hacking_dataset/'              # location of dataset

import numpy as np
import matplotlib.pyplot as plt
import pickle
import xgboost
import tensorflow as tf
import shap
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import utility_funcs as utf
import utils_chd as chd_utils

np.random.seed(10)
PLOT_GRAPHS = True # Adjust to True/False to enable/disable plotting of graphs, this can block execution if set to True
tf.random.set_seed(10)


def load_data():
    data_chd = None
    try:
        data_chd = pickle.load(open("{}/data_chd.pkl".format(save_loc), "rb"))
        print("Data found, loading in preprocessed dataset...")
    except:
        print("Data not found, reading in dataset...")

    if(data_chd is None):
        data_chd = chd_utils.load_and_preprocess_chd() 
        pickle.dump(data_chd, open("{}/data_chd.pkl".format(save_loc), "wb"))
    return data_chd

def XGBoost_train(data_chd):
    XGBoost = None
    try:
        XGBoost = pickle.load(open("{}/XGBoost.pkl".format(save_loc), "rb"))
        print("XGBoost model found, loading...")
    except:
        print("XGBoost model not found, training a new one...")

    if(XGBoost is None):
        XGBoost = xgboost.XGBClassifier(objective="binary:logistic", seed=10)
        XGBoost.fit(data_chd['X_train'], data_chd['Y_train'])
        pickle.dump(XGBoost, open("{}/XGBoost.pkl".format(save_loc), "wb"))
    return XGBoost

def compute_model_performance(XGBoost, data_chd):
    results_model = {}
    try:
        results_model = pickle.load(open("{}/results.pkl".format(save_loc), "rb"))
    except:
        print("Results not found, computing model performance...")

    if(results_model == {}):
        results_model['y_pred_train'] = XGBoost.predict(data_chd['X_train'])
        results_model['y_pred_test'] = XGBoost.predict(data_chd['X_test'])
        
        results_model['performance_model_train'] = utf.compute_performance_stats(data_chd['Y_train'], results_model['y_pred_train'])
        results_model['performance_model_test'] = utf.compute_performance_stats(data_chd['Y_test'], results_model['y_pred_test'])
        pickle.dump(results_model, open("{}/results.pkl".format(save_loc), "wb"))
    return results_model

def compute_shap(data_chd, model, results_model):
    print("Starting SHAP value computation...")
    results_AE_SHAP = None
    history_AE_SHAP = None
    autoencoder_shap = None
    try:
        results_AE_SHAP, history_AE_SHAP = pickle.load(open("{}/data_1.pkl".format(save_loc), "rb"))
        autoencoder_shap = utf.Autoencoder(results_AE_SHAP['x_data'].shape[1], results_AE_SHAP['best_params'][2], results_AE_SHAP['best_params'][3]) # create the AE model object
        autoencoder_shap.full.load_weights('{}/AE_shap.weights.h5'.format(save_loc));  # -> estava dando erro aqui
    except:
        print("SHAP/Autoencoder results not found, computing SHAP values and training Autoencoder...")

    if (results_AE_SHAP is None):
        results_AE_SHAP = {}
        explainer = shap.TreeExplainer(model, data_chd['X_train'], feature_perturbation = "interventional", model_output='probability') # NB output='probability' decomposes inputs among Pr(Y=1='Attack'|X)
        results_AE_SHAP['shap_train'] = explainer.shap_values(data_chd['X_train'])
        results_AE_SHAP['shap_test'] = explainer.shap_values(data_chd['X_test'])


        # scale data using Sklearn minmax scaler
        results_AE_SHAP['scaler'] = MinMaxScaler(feature_range=(0, 1))                                     
        results_AE_SHAP['shap_train_scaled'] = results_AE_SHAP['scaler'].fit_transform(results_AE_SHAP['shap_train'])  # scale the training set data
        results_AE_SHAP['shap_test_scaled'] = results_AE_SHAP['scaler'].transform(results_AE_SHAP['shap_test'])        # scale the test set data

        shap_train_normal_only = results_AE_SHAP['shap_train_scaled'][data_chd['train_normal_locs']]
        results_AE_SHAP['x_data'], results_AE_SHAP['val_data'] = train_test_split(shap_train_normal_only, test_size=0.2, random_state=10)

        # perform grid search to find the best paramters to use for the autoencoder model
        # specify the paramters of the grid space to serach, i.e. can use: np.arange(448,800,4).tolist()
    
    results_AE_SHAP['parameters'] = {
                                    # encoder params to search across
                                    'dense_1_units':[64],                                   
                                    'dense_1_activation':['relu'],
                                    'dense_2_units':[32],                                 
                                    'dense_2_activation':['relu'],
                                    'dense_3_units':[16],                                           
                                    'dense_3_activation':['relu'],
                                    # decoder params to search across
                                    'dense_4_units':[32],                                        
                                    'dense_4_activation':['relu'],
                                    'dense_5_units':[64],                                    
                                    'dense_5_activation':['relu'],
                                    'dense_6_units':[results_AE_SHAP['x_data'].shape[1]],               
                                    'dense_6_activation':['sigmoid']    
                                }

    # perform the grid search and return parameters of the best model
    results_AE_SHAP['grid_search'], results_AE_SHAP['best_params'] = utf.get_hyper_Autoencoder(results_AE_SHAP['parameters'], results_AE_SHAP['x_data'], results_AE_SHAP['val_data'],
                                                                                            method='exact', num_epochs=10, batch_size=2048, AE_type = 'joint')   
    # Using the best parameters, build and train the final model
    autoencoder_shap = utf.Autoencoder(results_AE_SHAP['x_data'].shape[1], results_AE_SHAP['best_params'][2], results_AE_SHAP['best_params'][3], AE_type='joint') # create the AE model object

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=2, mode='min', restore_best_weights=True)     # set up early stop criteria
    history_AE_SHAP = autoencoder_shap.full.fit(results_AE_SHAP['x_data'], results_AE_SHAP['x_data'], epochs=1000, batch_size=2048, shuffle=True, validation_data=(results_AE_SHAP['val_data'],   
                                results_AE_SHAP['val_data']), verbose=2, callbacks=[early_stop]).history
    # plot the training curve
    plt.plot(history_AE_SHAP["loss"], label="Training Loss")
    plt.plot(history_AE_SHAP["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    # perform anomaly detection based on the reconstruction error of the AE and save results
    results_AE_SHAP['performance_new_attacks'], results_AE_SHAP['new_attack_pred_locs'], results_AE_SHAP['AE_threshold'] = chd_utils.AE_anomaly_detection(autoencoder_shap, results_AE_SHAP['shap_train_scaled'], 
                                                                                                                                                    results_AE_SHAP['shap_test_scaled'], data_chd['Y_test'])

    # # calculate overall accuracy of the IDS system (XGBoost IDS and Anomaly detector) to detect attacks new or old attacks on the NSL-KDD Testset+
    results_AE_SHAP['all_attack_pred_locs'] = np.unique(np.concatenate((results_AE_SHAP['new_attack_pred_locs'], np.where(results_model['y_pred_test']==1)[0] )))
    results_AE_SHAP['y_pred_all'] = np.zeros(len(data_chd['Y_test']),)
    results_AE_SHAP['y_pred_all'][results_AE_SHAP['all_attack_pred_locs']] = 1
    results_AE_SHAP['performance_overall'] = utf.compute_performance_stats(data_chd['Y_test'], results_AE_SHAP['y_pred_all'])

    # SAVE DATA
    pickle.dump(results_model, open("{}/results.pkl".format(save_loc), "wb")) 
    pickle.dump([results_AE_SHAP, history_AE_SHAP], open("{}/data_1.pkl".format(save_loc), "wb")) 
    try:
        autoencoder_shap.full.save_weights('{}/AE_shap.weights.h5'.format(save_loc))
    except Exception as e:
        print("Error saving autoencoder weights.", e)
        pass

def main():
    data_chd = load_data()
    model = XGBoost_train(data_chd)
    results_model = compute_model_performance(model, data_chd)
    compute_shap(data_chd, model, results_model)
    # Future work: work on PCA analysis

    
if __name__ == "__main__":
    main()

