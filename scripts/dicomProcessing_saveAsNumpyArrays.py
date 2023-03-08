import pydicom as pdcm
import numpy as np
import os

from dicomProcessing_helpingFunctions import *


pos_path = 'D:/CT-Dataset/positive'
neg_path = 'D:/CT-Dataset/negative'
main_path = 'C:/Users/vikis/iCloudDrive/Škola/Matfyz/Diplomová Práca/dataset/Škola data/pozitivne'
ct_scans = list()
count = 1

print('Starting...')
for folder in os.listdir(main_path):
    path = main_path + '/' + str(folder)

    # Check if patient is positive or negative
    json_file = list(filter(lambda x: '.json' in x.lower(), os.listdir(path)))
    assert len(json_file) == 1

    df = pd.read_json(path + '/' + str(json_file[0]))
    d = pd.json_normalize(df.data)
    d = d[d['text'] == 'leukoencefalopatia']
    isPos = d['value'].values[0]

    if isPos:
        patient_path = pos_path + '/' + str(folder)
    else:
        patient_path = neg_path + '/' + str(folder)
    
    # Directory might contain other file(s) such as .json etc.
    files = list(filter(lambda x: '.dcm' in x.lower(), os.listdir(path)))
    for file in files:
        data = pdcm.dcmread(path + '/' + str(file))

        # Type of wanted description "Head  3.0  MPR"
        if data.SeriesDescription in allowed_types():
            ct_scans.append((data.pixel_array.tolist(), data.InstanceNumber))

    # CT Scans are saved in random order
    ct_scans.sort(key=lambda tup: tup[1])
    ct_scans = np.array([img for img, index in ct_scans])

    
    os.mkdir(patient_path)
    np.save(patient_path + '/numpy_data_raw.npy', ct_scans)
    
    print(count, end='-> ')
    count += 1
