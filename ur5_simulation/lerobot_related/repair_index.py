import pandas as pd
from os import listdir, environ
from os.path import isfile, join
import pyarrow as pa
import pyarrow.parquet as pq

mypath = environ['HOME'] + '/training_data/lerobot/my_pusht/data/chunk-000'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()
#print(onlyfiles)

'''
for file in onlyfiles:
    df = pd.read_parquet(mypath + '/' + file)
    print(f"file:{file} first index:{df['index'][0]}, last index:{df['index'][len(df)-1]}")
'''

#'''
for i in range(len(onlyfiles)-1):
    df1 = pd.read_parquet(mypath + '/' + onlyfiles[i])
    df2 = pd.read_parquet(mypath + '/' + onlyfiles[i+1])
    last_index_of_df1 = df1['index'][len(df1)-1]
    first_index_of_df2 = df2['index'][0]
    if first_index_of_df2 == last_index_of_df1 + 1:
        print(f'OK:former{last_index_of_df1}, latter{first_index_of_df2}')
    else:
        print(f'inconsistency detected:former{last_index_of_df1}, latter{first_index_of_df2}')
        for j in range(len(df2)):
            #print(f"prev:{df2.loc[j, 'index']}")
            df2.loc[j, 'index'] = df1['index'][len(df1)-1]+1+j
            #print(f"post:{df2.loc[j, 'index']}")
        table = pa.Table.from_pandas(df2)
        pq.write_table(table, mypath + '/' + onlyfiles[i+1])
        #print(f"repaired file:{onlyfiles[i+1]}")
    print(f"{df2}")
#'''