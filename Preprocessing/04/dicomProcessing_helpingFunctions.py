import os
import pydicom as pdcm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import skimage
import cv2


def normalize(data, k=1):
    """
    Fuction normalizes given data to min-max scale
    
    :param data: Numpy array data
    :return: 0-k scaled numpy array data
    """
    assert type(data) == np.ndarray
    return k*(data - np.min(data))/(np.max(data) - np.min(data))


def transform_to_hu(dicom_data, slope=None, intercept=None):
    """
    Function transforms DICOM data (Digital Imaging and Communication in Medicine)
    format to Hounsfield Unit (HU) given relationship
    data_HU = data*slope + intercept

    :param dicom_data: Data in DICOM format
    :param slope: Slope value
    :param intercept: Intercept value
    :return: pixel array in HU
    """
    if slope is None:
        slope = dicom_data.RescaleSlope

    if intercept is None:
        intercept = dicom_data.RescaleIntercept

    return dicom_data.pixel_array * slope + intercept


def filter_data(data, show_dif=False):
    """
    Function sets every pixel value beyond given threshold to 0 (default backgrond value)
    Thresholds are listed below in the table
    Afterwards using erosion and dilation redundant parts are removed (such as ears e.g.)

    ------------------------------------------------
    - Table values are based on data normalization -
    ------------------------------------------------
    | Bone     80  (Max)   | Brain   +-50          |
    | Vacuum   0   (Min)   |                       |
    | Water    18          |                       |
    ------------------------------------------------

    :param data: CT scan data array in HU
    :param threshold: Threshold beyond to filter out
    :param show_diff: True/False (debug parameter) - if true result is shown
    :return: Filtered out data
    """
    filter_values = {'bone': 80, 'vacuum': 0, 'water': 18, 'brain': 50}
    org_scan = data.copy()
    data[data >= filter_values['bone']] = 0
    data[data <= filter_values['water']] = 0
    
    labels, label_nb = sp.ndimage.label(data)
    label_count = np.bincount(labels.ravel().astype(int))
    label_count[0] = 0

    mask = labels == label_count.argmax()

    # mask = skimage.morphology.erosion(mask, np.ones((3, 3)))
    mask = skimage.morphology.erosion(mask, np.ones((4, 4)))

    # @TODO - dilataciu mozno prec
    mask = skimage.morphology.dilation(mask, np.ones((3, 3)))

    res = mask * data
    
    if show_dif:
        fig, ax = plt.subplots(1, 2, figsize=[8, 8])
        ax[0].set_title('Original scan')
        ax[0].imshow(org_scan, aspect='auto', cmap=plt.cm.bone)
        
        ax[1].set_title('Filtered out scan')
        ax[1].imshow(res, aspect='auto', cmap=plt.cm.bone)
        plt.show(block=True)
    
    return res 


def clear_data(data, dil_mat=(10, 10), show_diff=False):
    """
    Function filters out noise in scan, it sets all
    pixels > 100 (bone) to 100 and all
    pixels < 0 (background) to value 0
    (original scan max value is approx. 1000 and min value -1000)
    
    :param data: CT scan data array in HU
    :param show_diff: True/False (debug parameter) - if true result is shown
    :return: clean (without noise) data
    """
    org_scan = data.copy()

    # Set pixel values range from 0 to 100
    data[data > 100] = 100
    data[data < 0] = 0
    
    segmentation = skimage.morphology.dilation(data, np.ones(dil_mat))
    
    labels, label_nb = sp.ndimage.label(segmentation)    
    label_count = np.bincount(labels.ravel().astype(int))
    label_count[0] = 0

    mask = labels == label_count.argmax()
    mask = skimage.morphology.dilation(mask, np.ones(dil_mat))
    mask = sp.ndimage.morphology.binary_fill_holes(mask)
    mask = skimage.morphology.dilation(mask, np.ones(dil_mat))

    data = cv2.erode(data, np.ones((4, 4)))
    
    if show_diff:
        fig, ax = plt.subplots(1, 2, figsize=[8, 8])
        ax[0].set_title('Noisy scan')
        ax[0].imshow(org_scan, aspect='auto', cmap=plt.cm.bone)
        
        ax[1].set_title('Clean scan')
        ax[1].imshow(mask * data, aspect='auto', cmap=plt.cm.bone)
        plt.show(block=True)
    return mask * data


def mirror_scan(data, return_org=False):
    """
    Function flips horizontally scan in order to avoid false positives
    based on different structures of brain hemispheres

    :param data: Single CT scan data array
    :return: Fliped data and original (if return_org == True)
    """
    if return_org:
        org_scan = data.copy()
        return cv2.flip(data, 1), org_scan
    return cv2.flip(data, 1)


def resize_scan(data, out_width=512, out_height=512, show_diff=False):
    """
    Function resizes given data array of size = org_width x org_height
    to new resolution = out_width x out_height
    
    :param data: Single CT scan data array
    :param out_width: Output data width
    :param out_height: Output data height
    :param show_diff: True/False (debug parameter) - if true result is shown
    :return: Resized data
    """
    new = skimage.transform.resize(data, (out_width, out_height), preserve_range=True, anti_aliasing=False)
    # new = skimage.transform.rescale(data, 0.25,  anti_aliasing=False)

    if show_diff:
        fig, ax = plt.subplots(1, 2, figsize=[8, 8])
        ax[0].set_title('Original scan')
        ax[0].imshow(data, aspect='auto', cmap=plt.cm.bone)
        
        ax[1].set_title('Resized scan')
        ax[1].imshow(new, aspect='auto', cmap=plt.cm.bone)
        plt.show(block=True)
    return new


def move_to_center(data, out_width=512, out_height=512, show_diff=False):
    """
    Function resizes CT scan to given resolution
    First it shifts brain in the center
    Additional padding is added afterwards

    :param data: Single CT scan data array
    :param out_width: Output data width
    :param out_height: Output data height
    :param show_diff: True/False (debug parameter) - if true result is shown
    :returns: Single CT scan shifted to center of array
    """
    # Brain is surrounded by black backgroud (black pixel = 0) 
    mask = data == 0

    # Brain_array = Scan_array - Scan_background 
    coords = np.array(np.nonzero(~mask))
    
    try:
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)
    except ValueError:
        # First slices of CT scan are just air (slice with only black pixels)
        if show_diff:
            print(data.shape)
            print(coords)
            plt.imshow(data, cmap='bone')
            plt.show()
        return data
    
    # Remove the background
    croped_scan = data[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    if croped_scan.shape[0] > out_width or croped_scan.shape[1] > out_height:
        croped_scan = resize_scan(croped_scan, out_width, out_height)

    height, width = croped_scan.shape
    final_image = np.zeros((out_height, out_width))

    pad_left = int((out_width - width) // 2)
    pad_top = int((out_height - height) // 2)
    
    
    # Replace the pixels with the image's pixels
    #try:
    final_image[pad_top:pad_top + height, pad_left:pad_left + width] = croped_scan
    # final_image.ravel([pad_top:pad_top + height, pad_left:pad_left + width])

    if show_diff:
        fig, ax = plt.subplots(1, 2, figsize=[8, 8])
        ax[0].set_title('Original scan')
        ax[0].imshow(data, aspect='auto', cmap=plt.cm.bone)
        
        ax[1].set_title('Shifted scan')
        ax[1].imshow(final_image, aspect='auto', cmap=plt.cm.bone)
        plt.show(block=True)

    return final_image


def allowed_types():
    """
    Function returns list of allowed CT scan series descriptions
    """
    return ['Head  3.0  MPR']

def rotate_to_center(data, show_diff=False):
    """
    Function straightens tilted brain scan

    :param data: Single tilted CT scan data array
    :param show_diff: True/False (debug parameter) - if true result is shown
    :returns: Straight CT scan data array
    """
    # @TODO - nefunguje vo vela pripadoch
    scan = np.uint8(data)
    contours, hier = cv2.findContours(scan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(scan.shape) #, np.uint8)

    # find the biggest contour (c) by the area
    c = max(contours, key=cv2.contourArea)

    (x, y), (MA, ma), angle = cv2.fitEllipse(c)

    cv2.ellipse(scan, ((x, y), (MA, ma), angle), color=(0, 255, 0), thickness=2)

    rmajor = max(MA, ma)/2

    if angle > 90:
        angle -= 90
    else:
        angle += 90

    X_top = x + np.cos(np.radians(angle)) * rmajor
    Y_top = y + np.sin(np.radians(angle)) * rmajor
    X_bot = x + np.cos(np.radians(angle + 180)) * rmajor
    Y_bot = y + np.sin(np.radians(angle + 180))* rmajor

    M = cv2.getRotationMatrix2D((x, y), angle - 90, 1)
    straight_scan = cv2.warpAffine(scan, M, (scan.shape[1], scan.shape[0]), cv2.INTER_CUBIC)

    if show_diff:
        # Show line of rotation
        cv2.line(scan, (int(X_top), int(Y_top)), (int(X_bot), int(Y_bot)), (0, 255, 0), 3)

        fig, ax = plt.subplots(1, 2, figsize=[8, 8])
        ax[0].set_title('Tilted scan')
        ax[0].imshow(scan, aspect='auto', cmap=plt.cm.bone)
        
        ax[1].set_title('Straight scan')
        ax[1].imshow(straight_scan, aspect='auto', cmap=plt.cm.bone)
        plt.show(block=True)
        
    return straight_scan


def uniform_depth(data, depth=30, center_pos=3/5, type_='multi'):
    """
    Function uniforms depth of CT scans
    (given size of the body there might be more/less scans)

    -----------------------------------------
    In case of leukoencefalopathy classification:
    Ventricular system is located approx. in 3/5 depth of the scan
    -----------------------------------------
    
    :param data: numpy array data
    :param depth: number of
    :param center_pos: location of desired object in %
    :type_: single/multi - single=single patient/ multi=multi patients
    :return: uniformed depth in data
    """
    if type_ == 'single':
        data = np.array([data])
    
    if type(data) == np.ndarray:
        data = data.tolist()


    ct_scans = list()
    for scan in data:
        # Take only middle slices of CT scan (first/last parts have no important information)
        # If CT scan has 71 and desired depth is 30 -> final CT scan = original_scan[20:50]
        center = int(len(scan)*center_pos)
        ct_scans.append(scan[center - (depth // 2): center + (depth // 2)])

    if type_ == 'single':
        return np.array(ct_scans[0])
    return np.array(ct_scans)



def plot_ct_images(data, rows=6, cols=6, start_with=10, show_every=3, block=True):
    fig, ax = plt.subplots(rows, cols, figsize=[8, 8])
    index = start_with
    for i in range(rows):
        for j in range(cols):
            ax[i, j].set_title(f'Scan slice {index}')
            ax[i, j].imshow(data[index], aspect='auto', cmap=plt.cm.bone)
            ax[i, j].axis('off')
            index += show_every
    plt.show(block=block)

def load_data(path, limit=1000, processed_type='numpy_data_processed'):
    """
    Function loads all data processed to numpy file(.npy)
    from all folders in a given path

    :param path: path to folder containing CT scans (pos or neg)
    :param limit: int limit how many files to return
    :return: CT scans in numpy array
    """
    all_ct_scans = list()
    count = 1
    print('Path:', path)
    print('Loading CT data from numpy arrays...')
    for directory in os.listdir(path):
        if count <= limit:
            dir_path = path + '/' + str(directory)
            try:
                ct_scan = np.load(dir_path + '/' + processed_type + '.npy', allow_pickle=True)
                if ct_scan.shape != (0,) and len(ct_scan) != 0 and ct_scan.shape[0] > 5:
                    all_ct_scans.append(ct_scan.tolist())
                    print('-', end='')
                    if count % 10 == 0:
                        print(f'> {count} DONE!')
                count += 1
            except FileNotFoundError:
                pass
        else:
            pass
    print('CT data successfully loaded!')
    

    return np.array(all_ct_scans)


def remove_background(data, show_diff=False):
    scan = np.uint8(data)
    contours, hier = cv2.findContours(scan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(scan.shape, np.uint8)

    # find the biggest contour (c) by the area
    c = max(contours, key=cv2.contourArea)
    (x, y), (MA, ma), angle = cv2.fitEllipse(c)

    # cv2.ellipse(scan, ((x, y), (MA, ma), angle), color=(3, 5, 255), thickness=2)
    cv2.ellipse(mask, ((x, y), (MA, ma), angle), 255, thickness=2)

    result = cv2.bitwise_and(mask, scan)
    result[mask == 0] = 1

    if show_diff:
        fig, ax = plt.subplots(1, 2, figsize=[8, 8])
        ax[0].set_title('Background scan')
        ax[0].imshow(data, aspect='auto', cmap=plt.cm.bone)
        
        ax[1].set_title('W/o background scan')
        ax[1].imshow(result, aspect='auto', cmap=plt.cm.bone)
        plt.show(block=True)

    return result
    

    
