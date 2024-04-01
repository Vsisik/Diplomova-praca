import pydicom as pdcm
import numpy as np
import os

path = 'C:/Users/vikis/iCloudDrive/Škola/Matfyz/Diplomová Práca/dataset/Škola data/pozitivne – filtrovane/Mozog_ID500_MozogCT_Anonymized_1.3.6.1.4.1.20468.2.152.0.1.137527.924903'
unique_types = set()
to_keep, to_delete = list(), list()


check = input('Do you really want to delete files? (y/n): ')

# Directory might contain other file(s) such as .json etc.
files = list(filter(lambda x: '.dcm' in x.lower(), os.listdir(path)))

for file in files:
    file_path = path + '/' + str(file)
    data = pdcm.dcmread(file_path)

    # Type of wanted description "Head  3.0  MPR"
    if data.SeriesDescription == "Head  3.0  MPR":
        to_keep.append(file_path)
        
    # Delete all other files
    else:
        to_delete.append(file_path)

if check:
    for file_path in to_delete:
        os.remove(file_path)


print(f'''--- Summary of directory ---
Total num. of files:/t/t{len(files)}
Total num. of deleted files:/t{len(to_delete)}
All types of series:/t/t{unique_types}''')


