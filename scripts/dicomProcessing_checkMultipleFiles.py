import pydicom as pdcm
import numpy as np
import os

from dicomProcessing_helpingFunctions import *

np.random.seed = 5

path = 'D:/CT-Dataset/all'
path = path + '/' + np.random.choice(os.listdir(path))
unique_types = set()
unique_nums = set()
ct_scans = list()
print(f'Patient file: {path}')

# Directory might contain other file(s) such as .json etc.
files = list(filter(lambda x: '.dcm' in x.lower(), os.listdir(path)))
for file in files:
    data = pdcm.dcmread(path + '/' + str(file))
    unique_types.add(data.SeriesDescription)

    # Type of wanted description "Head  3.0  MPR"
    if data.SeriesDescription in allowed_types():
        scan = transform_to_hu(data)
        scan = clear_data(scan, show_diff=True)
        # scan = remove_background(scan)
        scan = rotate_to_center(scan, show_diff=True)
        scan = move_to_center(scan, 512, 512)
        scan = resize_scan(scan, 128, 128, show_diff=True)
        scan = filter_data(scan, show_dif=True)
        
        ct_scans.append((scan.tolist(), data.InstanceNumber))
        

# CT Scans are saved in random order
ct_scans.sort(key=lambda tup: tup[1])

print(f'''--- Summary of directory ---
Total num. of files:\t\t{len(files)}
Total num. of selected files:\t{len(ct_scans)}
All types of series:\t\t{unique_types}''')

ct_scans = np.array([img for img, index in ct_scans])
#ct_scans = ArithmeticError
plot_ct_images(ct_scans, 4, 4, 25, 1)
