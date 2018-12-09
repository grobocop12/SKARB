#! /usr/bin/python
import pandas as pd

data = pd.read_csv('genfile_file4.csv', encoding= 'cp1250')

print(data.head())
