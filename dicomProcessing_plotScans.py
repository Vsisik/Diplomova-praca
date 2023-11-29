"""
This is part of helper functions focusing solely on plotting scans
using matplotlib library
"""
from typing import Any

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def plot_ct_images(data: Any,
                   rows: int = 6,
                   cols: int = 6,
                   start_with: int = 10,
                   show_every: int = 3,
                   block: bool = True,
                   ):
    """
    Using matplotlib plots scans side by side

    :param data: list of scans
    :param rows: number of rows
    :param cols: number of columns
    :param start_with: initial scan to start with
    :param show_every: number of steps
    :param block: boolean freeze code while showing the figure
    :return: True (if success)
    """
    fig, ax = plt.subplots(rows, cols, figsize=[8, 8])
    index = start_with
    for i in range(rows):
        for j in range(cols):
            ax[i, j].set_title(f'Scan slice {index}')
            ax[i, j].imshow(data[index], aspect='auto', cmap=plt.cm.bone)
            ax[i, j].axis('off')
            index += show_every
    plt.show(block=block)
    return True


def plot_ct_image(data: Any,
                  title: str=None,
                  save: bool=False,
                  save_path: str=None,
                  block: bool=True,
                  ):
    """
    Plot scan with pixels range next to it

    :param data: single scan
    :param title: plot title
    :param save: boolean save the figure
    :param save_path: output file path (only if save=True) defaults to current folder
    :param block: boolean freeze code while showing the figure
    :return: True (if success)
    """
    if data.ndim != 2:
        raise Warning("Wrong shape of data!")

    plt.imshow(data)
    plt.title(title if title is not None else "CT scan")
    plt.colorbar()

    if save:
        plt.savefig(save_path if not None else "CT_scan.png")

    plt.show(block=block)

    return True


def plot_difference(original_data: Any,
                    new_data: Any,
                    func_title: str,
                    block: bool=True):
    """

    :param original_data: original scan
    :param new_data: processed scan
    :param func_title: process function name
    :param block: boolean freeze code while showing the figure
    :return: True (if success)
    """
    fig, ax = plt.subplots(1, 2, figsize=[8, 8])
    ax[0].set_title(f'Before {func_title}')
    ax[0].imshow(original_data, aspect='auto', cmap=plt.cm.bone)

    ax[1].set_title(f'After {func_title}')
    ax[1].imshow(new_data, aspect='auto', cmap=plt.cm.bone)

    plt.show(block=True)
    return True