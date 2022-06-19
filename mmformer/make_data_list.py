import os
join = os.path.join
import numpy as np

data_path = './data/Training/'
data_list_file = './data/train.txt'

types = ['HGG', 'LGG']

data_list = []
for i in types:
    temp_list = os.listdir(join(data_path, i))
    temp_list = [i+'/'+x for x in temp_list]
    data_list += temp_list

data_list = np.sort(data_list)
with open(data_list_file, 'w') as f:
    for i in data_list:
        f.write(i+'\n')
