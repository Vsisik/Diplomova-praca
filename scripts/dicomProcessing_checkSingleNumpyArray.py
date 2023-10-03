import pydicom as pdcm
import numpy as np
import os

from dicomProcessing_helpingFunctions import *

np.random.seed(20)

main_path = 'D:/CT-Dataset'
pos_path = main_path + '/positive'
neg_path = main_path + '/negative'
c_load = 'numpy_data_raw.npy'
# c_save = 'numpy_data_processed.npy'


# what = str(input('POS/NEG file: '))
what = 'pos'
if what.lower() == 'pos':
    path = pos_path

if what.lower() == 'neg':
    path = neg_path

unique_types = set()
all_ct_scans = list()
show_scan = False
save = False

for directory in [np.random.choice(os.listdir(path))]:
    print(directory)
    dir_path = path + '/' + str(directory)
    ct_scans = list()

    # Load numpy array file (.npy)
    # all data are already transformed to HU
    numpy_array = np.load(dir_path + '/' + c_load, allow_pickle=True)

    # plot_ct_images(numpy_array, 4, 4, 1, 1, block=False)

    cts = uniform_depth(numpy_array, type_='single')
    # plot_ct_images(cts, 4, 4, 1, 1, block=False)

    # For each scan in CTs
    for ct in cts:

        # To save the results
        if save:
            mats = [(1, 1), (3, 3), (5, 5), (8, 8), (10, 10),
                    (13, 13), (15, 15), (18, 18), (20, 20), (25, 25)]

            for mat in mats:
                scan = clear_data(ct, dil_mat=mat, show_diff=False, save=save)

        scan = clear_data(ct, dil_mat=(5, 5), show_diff=False)
        # scan = remove_background(scan, show_diff=True)
        # scan = rotate_to_center(scan, show_diff=False)
        # scan = move_to_center(scan, 512, 512, show_diff=False)
        scan = resize_scan(scan, 256, 256, show_diff=True)
        scan = filter_data(scan, show_dif=True)
        # scan = move_to_center(scan, 128, 128, show_diff=True)
        # scan = rotate_to_center(scan, show_diff=True)
        
        ct_scans.append(scan.tolist())

    ct_scans = np.array(ct_scans)
    plot_ct_images(ct_scans, 4, 4, 1, 1, block=False)
