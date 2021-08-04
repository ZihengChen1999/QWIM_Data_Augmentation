__author__ = "Ziheng Chen"
__version__ = "1"
__status__ = "Prototype"  # Status should typically be one of "Prototype", "Development", or "Production".

import numpy as np
import pandas as pd
import os
import evaluation_model

#change the file path to your path saving the data
file_path=os.path.join("/Users","chenziheng","Desktop","BoA","Codes","QWIMProject")
stock_data_source="Six_stocks.xlsx"

stock_data=pd.read_excel(os.path.join(file_path,stock_data_source),engine='openpyxl')
#need to install "openpyxl" use "pip install openpyxl"


stock_data.set_index("Date",inplace=True)
stock_data.dropna(inplace=True)
stock_data.sort_index(ascending=True,inplace=True)

eval_model=evaluation_model.Evaluation_model(stock_data)# let true_data be input here
eval_model.train()


print(eval_model.predict(stock_data)) # put the data you want to test here



