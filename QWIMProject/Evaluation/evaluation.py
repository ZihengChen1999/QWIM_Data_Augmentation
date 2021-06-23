__author__ = "Ziheng Chen"
__version__ = "1"
__status__ = "Prototype"  # Status should typically be one of "Prototype", "Development", or "Production".

import pandas as pd
import os

file_path=os.path.join("/Users","chenziheng","Desktop","BoA","Codes","QWIMProject")
six_stocks_data_source="Six_stocks.xlsx"

six_stocks_data=pd.read_excel(os.path.join(file_path,six_stocks_data_source),engine='openpyxl')
#need to install "openpyxl" use "pip install openpyxl"

six_stocks_data.set_index("Date",inplace=True)
six_stocks_data.dropna(inplace=True)
six_stocks_data.sort_index(ascending=True,inplace=True)

print(six_stocks_data)