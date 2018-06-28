import pandas as pd
import numpy as np

df = pd.read_csv('1625Data.txt',header=None)
catalog = 'ARNDCQEGHILKMFPSTWYV'

df_train = pd.DataFrame(np.zeros((1625,120), dtype='uint8'))



protein_count = 0
for protein in df[0]:
    print(protein)
    letter_count = 0
    for letter in protein:
        print('protein count: ' + str(protein_count))
        print('letter count: ' + str(letter_count))
        print(catalog.find(letter))
        catalog_indice = catalog.find(letter)
        df_train.iloc[protein_count][letter_count*20 + catalog_indice] = 1
        letter_count += 1
    protein_count += 1 
            

# df_train.iloc[2:3,0:20] particular row
