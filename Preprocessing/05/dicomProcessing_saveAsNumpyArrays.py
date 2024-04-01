import pydicom as pdcm
import numpy as np
import os
import pandas as pd
import shutil
import traceback

from dicomProcessing_helpingFunctions import *


pos_path = 'D:/CT-Dataset/positive'
neg_path = 'D:/CT-Dataset/negative'
main_path = 'D:/CT-Dataset/all'
count = 1

print('Starting...')
for folder in os.listdir(main_path):
    ct_scans = list()
    path = main_path + '/' + str(folder)
    if not (os.path.exists(pos_path + '/' + str(folder)) or (os.path.exists(neg_path + '/' + str(folder)))):
        try:
            # -----------------------------------
            # Check for patient medical file
            json_file = list(filter(lambda x: '.json' in x.lower(), os.listdir(path)))
            try:
                assert len(json_file) == 1
            except AssertionError:
                json_file = list(filter(lambda x: all(ext not in x.lower() for ext in ['.dcm', '.txt', '.zip']), os.listdir(path)))
                assert len(json_file) == 1


            df = pd.read_json(path + '/' + str(json_file[0]))
            # -----------------------------------
            # Newer files contain header
            try:
                d = pd.json_normalize(df.data)
            # -----------------------------------
            # Older files have no header
            except AttributeError:
                d = df.copy()


            # -----------------------------------
            # Check if patient's postive/negative
            consistency_list = []
            parent = None
            if 'leukoencefalopatia' in d['text'].values:
                a = d[d['text'] == 'leukoencefalopatia']
                parent = a['value'].values[0]
                
            if '472717ac-e16c-4b5f-bedd-7801c93c3ca7' in d['guid'].values:
                a = d[d['guid'] == '472717ac-e16c-4b5f-bedd-7801c93c3ca7']
                consistency_list.append(a['value'].values[0])

            if '6591ee49-b0b4-4f85-8d22-2eb3e1c3306a' in d['guid'].values:
                a = d[d['guid'] == '6591ee49-b0b4-4f85-8d22-2eb3e1c3306a']
                consistency_list.append(a['value'].values[0])

            if '9fc5fc12-c313-43f9-8a9b-6dfb32b903ab' in d['guid'].values:
                a = d[d['guid'] == '9fc5fc12-c313-43f9-8a9b-6dfb32b903ab']
                consistency_list.append(a['value'].values[0])

            # -----------------------------------
            if parent is not None:
                isPos = parent

            else:
                isPos = any(consistency_list)


            # -----------------------------------
            if isPos:
                patient_path = pos_path + '/' + str(folder)
            else:
                patient_path = neg_path + '/' + str(folder)

            # Create new directory with name = patient ID
            os.mkdir(patient_path)
            
            # Directory might contain other file(s) such as .json etc.
            files = list(filter(lambda x: '.dcm' in x.lower(), os.listdir(path)))
            for file in files:
                data = pdcm.dcmread(path + '/' + str(file))


            # Type of wanted description "Head  3.0  MPR"
                if data.SeriesDescription in allowed_types():
                    scan = transform_to_hu(data)
                    ct_scans.append((scan.tolist(), data.InstanceNumber))


            # CT Scans are saved in random order
            ct_scans.sort(key=lambda tup: tup[1])
            ct_scans = np.array([img for img, index in ct_scans], dtype='object')


            np.save(patient_path + '/numpy_data_raw.npy', ct_scans)
        
            print(count, end='-> ')
            if count % 10 == 0:
                print()

        except AssertionError:
            # Json format file not found (prolly contains json without suffix)
            # print(path)
            pass
        
        except IndexError:
            # No text leuko... found in json
            traceback.print_exc()
            pass

        except FileExistsError:
            # File already exists
            try:
                shutil.rmtree(path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
            pass
        count += 1
