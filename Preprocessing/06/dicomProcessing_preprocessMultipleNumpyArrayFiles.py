import pydicom as pdcm
import numpy as np
import os
import traceback

from dicomProcessing_helpingFunctions import *

path = None
main_path = 'D:/CT-Dataset'
pos_path = main_path + '/positive'
neg_path = main_path + '/negative'
c_load = 'numpy_data_raw.npy'
c_save = 'numpy_data_processed6.npy'

unique_types = set()
all_ct_scans = list()
show_scan = False
count = 0

what = str(input('POS/NEG file: '))
if what.lower() == 'pos':
    path = pos_path

if what.lower() == 'neg':
    path = neg_path


for directory in os.listdir(path):
    dir_path = path + '/' + str(directory)
    ct_scan = list()
    try:
        # TODO - vratit
        if True: # not os.path.exists(dir_path + '/' + c_save):
            # Load numpy array file (.npy)
            # all data are already transformed to HU
            numpy_array = np.load(dir_path + '/' + c_load, allow_pickle=True)

            uniformed = uniform_depth(numpy_array, center_pos=3/5, type_='single', depth=40)
            uniformed = uniformed.astype(np.float32)

            for scan_raw in uniformed:
                scan = scan_raw
                scan = thresholding(scan, show_diff=False)
                # scan = mean_filter(scan, mat=(3, 3), show_diff=True)
                # scan = adjust_contrast(scan, low=0.15, high=0.65, show_diff=True)
                # scan = clear_data(scan, dil_mat=(3, 3), show_diff=True)
                # scan = move_to_center(scan, out_height=256, out_width=256, show_diff=True)
                scan = resize_scan(scan, out_width=256, out_height=256, show_diff=False)
                
                ct_scan.append(scan.tolist())

            # Sort based on sequence slice number
            ct_scan = np.array(ct_scan, dtype=np.float32)

            np.save(dir_path + '/' + c_save, ct_scan, allow_pickle=True)
        count += 1
        print(count, end='->\t')
        if count % 10 == 0:
            print()

    except Exception as error:
        traceback.print_exc()
        pass 
