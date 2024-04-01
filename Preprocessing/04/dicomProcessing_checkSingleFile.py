import pydicom as pdcm
import numpy as np
import matplotlib.pyplot as plt
import os

path = 'D:/CT-Dataset/all/Mozog_ID500_MozogAGCT_Anonymized_1.3.6.1.4.1.20468.2.152.0.1.89917.664402'
file = path + '/' + str(np.random.choice(os.listdir(path)))

# Type of wanted description "Head  3.0  MPR"
data = pdcm.dcmread(file)
print(f''' <--- File: {file} --->
Series number: {data.SeriesNumber}''')
