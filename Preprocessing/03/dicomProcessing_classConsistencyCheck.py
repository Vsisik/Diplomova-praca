# CONSISTENCY CHECK
import os
import pandas as pd
import traceback


main_path = 'D:/CT-Dataset/all'

count = 0

for folder in os.listdir(main_path):
    ct_scans = list()
    path = main_path + '/' + str(folder)
    try:
        # Check if patient is positive or negative
        json_file = list(filter(lambda x: '.json' in x.lower(), os.listdir(path)))
        try:
            assert len(json_file) == 1
        except AssertionError:
            json_file = list(filter(lambda x: all(ext not in x.lower() for ext in ['.dcm', '.txt', '.zip']), os.listdir(path)))
            assert len(json_file) == 1
            # print(path + '/' + str(json_file[0]))

        # json_file = list(filter(lambda x: '.dcm' not in x.lower(), os.listdir(path)))

        df = pd.read_json(path + '/' + str(json_file[0]))
        # Newer files contain header
        try:
            d = pd.json_normalize(df.data)

        # Older files have no header
        except AttributeError:
            d = df.copy()

        
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
        if parent is not None:
            if parent:
                assert True in consistency_list
            else:
                assert all(v == parent for v in consistency_list)
        else:
            pass

            
    except AssertionError:
        print(consistency_list)
        traceback.print_exc()
        print(main_path + '/' + folder)
        # raise AssertionError

        count += 1
