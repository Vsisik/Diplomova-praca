import pydicom as pdcm
import numpy as np
import os

from dicomProcessing_helpingFunctions import *

# np.random.seed(20)
# np.random.seed(8) # preprocessing type 5

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

np.random.seed(2)
for directory in [np.random.choice(os.listdir(path))]:
    print(directory)
    dir_path = path + '/' + str(directory)
    ct_scans = list()

    # Load numpy array file (.npy)
    # all data are already transformed to HU
    numpy_array = np.load(dir_path + '/' + c_load, allow_pickle=True)

    # plot_ct_images(numpy_array, 4, 4, 1, 1, block=False)

    cts = uniform_depth(numpy_array, type_='single', depth=30, center_pos=6/10)
    # plot_ct_images(cts, 3, 5, start_with=1, show_every=2, block=False)
    
    # For every third scan in CTs
    for ct in cts[::3]:
        # Preprocessing 5
        scan = thresholding(ct, show_diff=True)
        scan = mean_filter(scan, mat=(3, 3), show_diff=True)
        scan = adjust_contrast(scan, low=0.15, high=0.65, show_diff=True)
        # scan = median_filter(scan, mat=(5, 5), show_diff=True)
        scan = clear_data(scan, dil_mat=(3, 3), show_diff=True)
        scan = move_to_center(scan, out_height=512, out_width=512, show_diff=True)
        scan = resize_scan(scan,  out_width=256, out_height=256, show_diff=True)
        plot_ct_image(scan)



        
        ct_scans.append(scan.tolist())

    ct_scans = np.array(ct_scans)
    plot_ct_images(ct_scans, 4, 4, 1, 1, block=False)
