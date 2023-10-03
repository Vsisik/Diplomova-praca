import matplotlib.pyplot as plt
import numpy as np
import math
import cv2


folder_path = './Images/Erosion'

img1_name = 'erosion_mat(3, 3)'
img1 = cv2.resize(cv2.imread(f'{folder_path}/{img1_name}.png', cv2.IMREAD_GRAYSCALE),
                  (480, 480))

img2_name = 'erosion_mat(8, 8)'
img2 = cv2.resize(cv2.imread(f'{folder_path}/{img2_name}.png', cv2.IMREAD_GRAYSCALE),
                  (480, 480))

img3_name = 'erosion_mat(13, 13)'
img3 = cv2.resize(cv2.imread(f'{folder_path}/{img3_name}.png', cv2.IMREAD_GRAYSCALE),
                  (480, 480))

img4_name = 'erosion_mat(25, 25)'
img4 = cv2.resize(cv2.imread(f'{folder_path}/{img4_name}.png', cv2.IMREAD_GRAYSCALE),
                  (480, 480))


rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}
plt.rcParams.update(rc)
figure, axis = plt.subplots(2, 2)

axis[0, 0].imshow(img1, cmap=plt.cm.bone)
# axis[0, 0].set_title(img1_name)

axis[0, 1].imshow(img2, cmap=plt.cm.bone)
# axis[0, 1].set_title(img2_name)

axis[1, 0].imshow(img3, cmap=plt.cm.bone)
# axis[1, 0].set_title(img3_name)

axis[1, 1].imshow(img4, cmap=plt.cm.bone)
# axis[1, 1].set_title(img4_name)

# figure.tight_layout()
figure.subplots_adjust(hspace=0.3, wspace=0)


plt.show()
