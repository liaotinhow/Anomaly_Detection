import pandas as pd 
import tensorflow as tf 
from keras.layers import Input, Dense 
from keras.models import Model 
from sklearn.metrics import precision_recall_fscore_support 
import matplotlib.pyplot as plt 
import ipdb
import csv
import sys
import numpy as np
import pickle
import json

def get_stock_info(file_name):
    stock_info = {}
    with open(file_name, newline='',encoding="utf-8") as stock:
        rows = csv.reader(stock)
        for row in rows:
            if not row[1].isnumeric():
                continue
            if row[1] in stock_info:
                if float(row[2]) != 0:
                    stock_info[row[1]]["info"].append(row)
            else:
                stock_info[row[1]] = {}
                stock_info[row[1]]["info"] = []
    
    for key in stock_info.keys():
        stock_info[key]["info"].sort(key =lambda x:x[0])
    return stock_info

def get_stock_anomaly(stock_info, plot=False):
    for key in stock_info.keys():
        print(f'key::::::::::::::::::::::   {key}')
        data = []
        for i in range(len(stock_info[key]['info'])):
            data.append([stock_info[key]['info'][i][3]])
        data_tensor = tf.convert_to_tensor(np.array(data), dtype=tf.float32)
        
        input_dim = data_tensor.shape[1]
        encoding_dim = 10

        input_layer = Input(shape=(input_dim,)) 
        encoder = Dense(encoding_dim, activation='relu')(input_layer) 
        decoder = Dense(input_dim, activation='relu')(encoder) 
        autoencoder = Model(inputs=input_layer, outputs=decoder) 
        ipdb.set_trace()
        # Compile and fit the model 
        autoencoder.compile(optimizer='adam', loss='mse') 
        autoencoder.fit(data_tensor, data_tensor, epochs=50, 
                        batch_size=32, shuffle=True) 

        # Calculate the reconstruction error for each data point 
        reconstructions = autoencoder.predict(data_tensor) 
        mse = tf.reduce_mean(tf.square(data_tensor - reconstructions), 
                            axis=1) 
        anomaly_scores = pd.Series(mse.numpy(), name='anomaly_scores') 
        #anomaly_scores.index = data_converted.index 
        threshold = anomaly_scores.quantile(0.99) 
        anomalous = anomaly_scores > threshold 
        binary_labels = anomalous.astype(int) 
        precision, recall, f1_score, _ = precision_recall_fscore_support( 
                binary_labels, anomalous, average='binary') 
        #test = data_converted['value'].values 
        predictions = anomaly_scores.values
        stock_info[key]['anomaly'] = anomalous

        if plot:
            print("Precision: ", precision) 
            print("Recall: ", recall) 
            print("F1 Score: ", f1_score) 
            plt.figure(figsize=(16, 8)) 
            t = [i for i in range(data_tensor.shape[0])]
            plt.plot(np.array(t),
                    np.array(data_tensor) )
            plt.plot(np.array(t)[anomalous], 
                    np.array(data_tensor)[anomalous], 'ro') 
            plt.title(f'Anomaly Detection: stock {key}') 
            plt.xlabel('Time') 
            plt.ylabel('Value')
            plt.savefig(f'figure/{key}_anomaly.png') 
            # plt.show() 


if __name__ ==  "__main__":
    stock_file_name = sys.argv[1]
    stock_info = get_stock_info(stock_file_name)
    ipdb.set_trace()
    get_stock_anomaly(stock_info, plot=True)
    ipdb.set_trace()
    with open("stock_info.pkl", "wb") as ff:
        pickle.dump(stock_info, ff)
    
    ipdb.set_trace()