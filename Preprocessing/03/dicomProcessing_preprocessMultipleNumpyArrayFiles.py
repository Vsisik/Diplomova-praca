import pydicom as pdcm
import numpy as np
import os
import traceback

from dicomProcessing_helpingFunctions import *


main_path = 'D:/CT-Dataset'
pos_path = main_path + '/positive'
neg_path = main_path + '/negative'
c_load = 'numpy_data_raw.npy'
c_save = 'numpy_data_processed3.npy'


what = str(input('POS/NEG file: '))
if what.lower() == 'pos':
    path = pos_path

if what.lower() == 'neg':
    path = neg_path

unique_types = set()
all_ct_scans = list()
show_scan = False
count = 0

for directory in os.listdir(path):
    dir_path = path + '/' + str(directory)
    ct_scan = list()
    try:
        if not os.path.exists(dir_path + '/' + c_save):
            # Load numpy array file (.npy)
            # all data are already transformed to HU
            numpy_array = np.load(dir_path + '/' + c_load, allow_pickle=True)

            uniformed = uniform_depth(numpy_array, center_pos=1/2, type_='single')

            for scan_raw in uniformed:
                scan = clear_data(scan_raw, dil_mat=(20, 20), show_diff=False)
                # scan = remove_background(scan, show_diff=False)
                scan = rotate_to_center(scan, show_diff=False)
                scan = move_to_center(scan, 512, 512, show_diff=False)
                scan = resize_scan(scan, 256, 256, show_diff=False)
                scan = filter_data(scan, show_dif=False)
                scan = move_to_center(scan, 128, 128, show_diff=False)
                # scan = rotate_to_center(scan, show_diff=True)
                
                ct_scan.append(scan.tolist())

            # Sort based on sequence slice number
            ct_scan = np.array(ct_scan)
            # ct_scan = normalize(ct_scan)

            np.save(dir_path + '/' + c_save, ct_scan, allow_pickle=True)
        count += 1
        print(count, end='->\t')
        if count % 10 == 0:
            print()
    except Exception as error:
        traceback.print_exc()
        pass 
