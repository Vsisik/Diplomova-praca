import pydicom as pdcm
import numpy as np
import os

from dicomProcessing_helpingFunctions import *

pos_path = 'C:/Users/vikis/iCloudDrive/Škola/Matfyz/Diplomová Práca/dataset/Škola data/pozitivne'
neg_path = 'C:/Users/vikis/iCloudDrive/Škola/Matfyz/Diplomová Práca/dataset/Škola data/negativne'

what = str(input('POS/NEG file: '))
if what.lower() == 'pos':
    path = pos_path

if what.lower() == 'neg':
    path = neg_path

unique_types = set()
all_ct_scans = list()
min_best = 10**4
show_scan = False

for directory in os.listdir(path):
    dir_path = path + '/' + str(directory)
    ct_scan = list()

    # Directory might contain other file(s) such as .json etc.
    files = list(filter(lambda x: '.dcm' in x.lower(), os.listdir(dir_path)))
    for file in files:
        data = pdcm.dcmread(dir_path + '/' + str(file))
        unique_types.add(data.SeriesDescription)

        # Type of wanted description "Head  3.0  MPR"
        if data.SeriesDescription in allowed_types():
            scan = transform_to_hu(data)
            scan = clear_data(scan)
            scan = rotate_to_center(scan)
            scan = move_to_center(scan, 512, 512)
            scan = resize_scan(scan, 128, 128)

            ct_scan.append((scan.tolist(), data.InstanceNumber))

    # Sort based on sequence slice number
    ct_scan.sort(key=lambda tup: tup[1])
    ct_scan = np.array([img for img, index in ct_scan])
    ct_scan = normalize(ct_scan)

    np.save(dir_path + '/numpy_data.npy', ct_scan)
    
    all_ct_scans.append(ct_scan.tolist())

    print(f'''--- Summary of directory ---
    Total num. of files:\t\t{len(files)}
    Total num. of selected files:\t{len(ct_scan)}
    All types of series:\t\t{unique_types}\n\n''')

all_ct_scans = uniform_depth(all_ct_scans, center_pos=3/5)
# all_ct_scans = normalize(all_ct_scans)

if show_scan:
    for scan in all_ct_scans:
        plot_ct_images(scan, 5, 5, 1, 1)
