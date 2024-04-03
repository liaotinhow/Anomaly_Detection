import os
import ipdb
import shutil
for spot in os.listdir('csv'):
    for stock_file in os.listdir(f'csv/{spot}'):
        if int(stock_file.split('_')[2]) >= 2:
            #ipdb.set_trace()
            try:
                os.mkdir(f"anomaly_csv/{stock_file.split('_')[2]}")
            except:
                pass
            dst = f"anomaly_csv/{stock_file.split('_')[2]}/spot_{spot}_stock_{stock_file.split('_')[0]}.tsv"
            shutil.copyfile(f'csv/{spot}/{stock_file}', dst)
            #with open(f'anomaly_csv/spot_{spot}_stock_{stock_file.split('_')[0]}.tsv', 'w') as f:

ipdb.set_trace()