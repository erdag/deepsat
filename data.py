import pandas as pd
import numpy as np

df = pd.read_csv('1625Data.txt',header=None)

print(df.head())

catalog = 'ARNDCQEGHILKMFPSTWYV'



def vectorize(protein):
    result = np.zeros((1,20),dtype='uint8')
    for p in protein:
        
        indice = catalog.find(p)
        result[0][indice] = 1
    return result




df['train_data'] = df[0].apply(vectorize)

print(df['train_data'][0])
print(df['train_data'][1])
print(df['train_data'][2])
print(df['train_data'][3])
