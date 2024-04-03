import ipdb
import csv
import sys
import os
stock_info = {}
with open(sys.argv[1], newline='') as stock:
    rows = csv.reader(stock)
    for row in rows:
        if row[1] in stock_info:
            stock_info[row[1]].append(row)
        else:
            stock_info[row[1]] = []
    
for key in stock_info.keys():
    stock_info[key].sort(key =lambda x:x[0])
ipdb.set_trace()