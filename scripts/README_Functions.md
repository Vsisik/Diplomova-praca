## **Funkcie**

### *Transform_to_hu*
Function transforms DICOM data (Digital Imaging and Communication in Medicine)
format to Hounsfield Unit (HU) using equation *data_HU* *=* *data * slope + intercept*

```python
transform_to_hu(dicom_data, slope=None, intercept=None)
```


***returns*** *numpy pixel array*

| **Parameter**        | **Type** | **Description**      |
| :------------------- | :------- | :------------------- |
| `dicom_data`         | `DICOM`  | Data in DICOM format |
| `slope`              | `Float`  | Slope value          |
| `intercept`          | `Float`  | Intercept value      |

\
\

***Example***
```python
>>> data = pdcm.dcmread(path_to_dicom_file)
>>> transform_to_hu(data)
array([[ -998.,  -992.,  -991., ..., -1008.,  -996.,  -984.],
       [-1003.,  -996.,  -990., ...,  -999.,  -979.,  -978.],
       [-1003., -1003., -1001., ...,  -988.,  -978.,  -990.],
       ...,
       [ -997.,  -998.,  -999., ..., -1003.,  -997.,  -997.],
       [ -996.,  -997.,  -999., ..., -1008., -1006., -1001.],
       [ -997.,  -997., -1001., ..., -1006., -1012., -1008.]])
