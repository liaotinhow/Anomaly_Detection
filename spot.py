import ipdb
import csv
import sys
import os
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
from datetime import date
import datetime
import argparse
'''
spot_dict = {}
for file in os.listdir(sys.argv[1]):
    with open(os.path.join(sys.argv[1],file)) as fd:
        data = csv.reader(fd, delimiter='\t')
        for line in data:
            if line[2] not in spot_dict:
                spot_dict[line[2]] = {}
            if line[1] not in spot_dict[line[2]]:
                spot_dict[line[2]][line[1]] = []
            spot_dict[line[2]][line[1]].append([line[0], line[3]])
            #print(line)
        fd.close()
ipdb.set_trace()
'''


def in_n_days(date1, date2, n_days):
    date1 = date(int(date1.split('-')[0]), int(date1.split('-')[1]), int(date1.split('-')[2]))
    date2 = date(int(date2.split('-')[0]), int(date2.split('-')[1]), int(date2.split('-')[2]))
    if date1 > date2:
        return False
    return date2-date1 < datetime.timedelta(days=n_days)

def find_correspond_anomaly_of_spot_and_stock(spot_per_company_anomaly, stock_anomaly_set, n_days):
    correspond_spot_anamoly = []
    correspond_stock = []
    for spot_ano in spot_per_company_anomaly:
        for stock_ano in stock_anomaly_set:
            if in_n_days(spot_ano[0], stock_ano[0], n_days):
                correspond_spot_anamoly.append(spot_ano)
                correspond_stock.append(stock_ano)
                #ipdb.set_trace()
    return correspond_spot_anamoly, correspond_stock

def find_stock_rise_or_fall_of_spot(spot_per_company_anomaly, stock_info, n_days, percentage):
    spot_anamoly_day_set = {}
    correspond_spot_anamoly = []

    # to do: check the day exists or not
    for spot_anamoly in spot_per_company_anomaly:
        spot_anamoly_day_set[spot_anamoly[0]] = spot_anamoly
    for i, stock_day_info in enumerate(stock_info):
        if stock_day_info[0] in spot_anamoly_day_set.keys():
            next = max(i+n_days, len(stock_info)-1)
            per = (float(stock_info[next][2]) - float(stock_info[i][2])) / float(stock_info[i][2])
            if per > 0 and per >= (percentage/100) and float(spot_anamoly_day_set[stock_day_info[0]][1]) > 0:
                correspond_spot_anamoly.append(spot_anamoly_day_set[stock_day_info[0]])
            if per < 0 and -per >= (percentage/100) and float(spot_anamoly_day_set[stock_day_info[0]][1]) < 0:
                correspond_spot_anamoly.append(spot_anamoly_day_set[stock_day_info[0]])
    return correspond_spot_anamoly


def save_spot_per_company_anamoly(spot_name, key, info):
    if not os.path.isdir(f"spot_per_company_anamoly"):
        os.mkdir(f"spot_per_company_anamoly")
    with open(f"spot_per_company_anamoly/{spot_name}_{key}_anamoly.pkl", 'wb') as f:
        pickle.dump(info, f)
    f.close()
    return 


def get_spot_anomaly(spot_name, spot_info, stock_anomaly, stock_info, args, plot=False):
    #if os.path.isdir(f'csv/{spot_name}'):
    #    return
    for key in spot_info.keys():
        try:
            if key in stock_anomaly:
                if os.path.exists(f"spot_per_company_anamoly/{spot_name}_{key}_anamoly.pkl"):
                    with open(f"spot_per_company_anamoly/{spot_name}_{key}_anamoly.pkl", 'rb') as f:
                        spot_per_company_anomaly = pickle.load(f)
                        f.close()
                else: 
                    stock_anomaly_set = [info for info in stock_anomaly[key]]
                    print(f'key::::::::::::::::::::::   {key}')
                    spot_per_company = sorted(spot_info[key])
                    
                    data = []
                    for i in range(len(spot_per_company)):
                        data.append([spot_per_company[i][1]])
                    data_tensor = tf.convert_to_tensor(np.array(data), dtype=tf.float32)
                    
                    input_dim = data_tensor.shape[1]
                    encoding_dim = 10

                    input_layer = Input(shape=(input_dim,)) 
                    encoder = Dense(encoding_dim, activation='relu')(input_layer) 
                    decoder = Dense(input_dim, activation='relu')(encoder) 
                    autoencoder = Model(inputs=input_layer, outputs=decoder) 

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
                    threshold = anomaly_scores.quantile(args.anamoly_threshold/100)
                    anomalous = anomaly_scores > threshold 
                    binary_labels = anomalous.astype(int) 
                    precision, recall, f1_score, _ = precision_recall_fscore_support( 
                            binary_labels, anomalous, average='binary') 
                    #test = data_converted['value'].values 
                    predictions = anomaly_scores.values
                    spot_per_company_anomaly = []

                    #print(spot_info)
                    anomaly_index = np.argwhere(anomalous==True).squeeze().tolist()
                    if isinstance(anomaly_index, int):
                        anomaly_index = [anomaly_index]
                    for i in anomaly_index:
                        spot_per_company_anomaly.append(spot_per_company[i])

                    save_spot_per_company_anamoly(spot_name, key, spot_per_company_anomaly)

                if args.cmd == 'find_stock_anomaly':
                    correspond_spot, correspond_stock = find_correspond_anomaly_of_spot_and_stock(spot_per_company_anomaly, stock_anomaly_set, args.days)
                    try:
                        os.mkdir(f'csv/{spot_name}')
                    except:
                        pass
                    #ipdb.set_trace()
                    correspond_file = f'csv/{spot_name}/{key}_with_{len(correspond_spot)}_corresponds.tsv'
                    if len(correspond_spot) >= 1:
                        with open(correspond_file, 'w', encoding='utf8', newline='') as tsvfile:
                            tsv_writer  = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
                            tsv_writer.writerow(["買賣時間", "買賣超", "股票日期", "股票代號", "收盤價", "漲跌幅(%)", "成交量", "成交值(百萬)"])
                            for i in range(len(correspond_spot)):
                                row = correspond_spot[i] + correspond_stock[i]
                                tsv_writer.writerow(row)
                        tsvfile.close()


                elif args.cmd == 'find_stock_rise_or_fall':
                    correspond_spot = find_stock_rise_or_fall_of_spot(spot_per_company_anomaly, stock_info[key]['info'], args.days, args.percentage)
                
                if len(correspond_spot) >= 1:
                    ipdb.set_trace()


                # plot spot per stock anomaly detection figure
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
                    try:
                        os.mkdir(f'spot_fig/{spot_name}')
                    except:
                        pass
                    plt.savefig(f'spot_fig/{spot_name}/{key}_anomaly.png') 
                    plt.close()
        except:
            pass
            

def get_stock_anomaly_info(stock_info):
    stock_anomaly = {}
    for key in stock_info.keys():
        stock_anomaly[key] = []
        for i in np.argwhere(stock_info[key]['anomaly']):
            stock_anomaly[key].append(stock_info[key]['info'][i[0]])
        stock_anomaly[key] = sorted(stock_anomaly[key])
    return stock_anomaly

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(prog='spot.py')
    subps = parser.add_subparsers(dest="cmd")

    stock_ana = subps.add_parser("find_stock_anomaly", help="find correspond stock anamoly with spot anamoly by stock")
    stock_ana.add_argument("--anamoly_threshold", type=float, default=99, help="anamoly threshold")
    stock_ana.add_argument("--days", default=30, help= "days of obersevation")

    rof = subps.add_parser("find_stock_rise_or_fall", help="find stock rise or fall over x percentage in y days")
    rof.add_argument("--anamoly_threshold", type=float, default=99, help="anamoly threshold")
    rof.add_argument("--days", type=int, default=30, help= "days of obersevation")
    rof.add_argument("--percentage", type=float, default=10, help= "stock rise or fall percentage")

    args = parser.parse_args()
    with open('stock_info.pkl', 'rb') as f:
        stock_info = pickle.load(f)
    f.close()
    stock_anomaly = get_stock_anomaly_info(stock_info)

    for spot_pkl in os.listdir('spots'):
        with open(os.path.join('spots', spot_pkl), 'rb') as f:
            spot = pickle.load(f)
            spot_name = spot_pkl.split('.')[0]
            get_spot_anomaly(spot_name, spot, stock_anomaly, stock_info, args)