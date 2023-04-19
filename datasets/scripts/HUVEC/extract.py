import importlib
import confocal
importlib.reload(confocal)

import glob, os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import imageio
import re
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import interp2d,RegularGridInterpolator
from random import random
from tqdm import tqdm
from scipy.interpolate import interp1d

from itertools import accumulate
from skimage.measure import find_contours
from scipy.spatial import ConvexHull
import time

import itertools

cell_numbers = [1, 2, 3, 4, 5, 6, 7, ]
raise Exception('Inspect this code before use on HUVECs; the directories have been changed')
base_dir = 'datasets/Parent/'
specific_ds_base = 'Parent Cell %d/'
ome_fname_base = 'Parent Cell %d_Reconstructed_crop.ome.tif'

n_channels = 3 # number of channels in ome (sometimes there's 4)
keep3 = [0, 1, 2] # needs to be adjusted when there's 4 channels
use_masks = False # takes time to do this using CellProfiler
plot_max = False # can be useful debugger but interupts running


def main():
    if True:
        for _ in tqdm(cell_numbers):  # 1,2,3,4
            base_dir = base_dir
            specific_ds = specific_ds_base % _
            target_dir = os.path.join(base_dir, specific_ds, 'Intensity.png.export/')
            source_folder = os.path.join(base_dir, specific_ds)

            #     print('isfile', os.path.isfile(os.path.join(source_folder,'HUVEC Cell 1.ome.tif')))

            ome_fname = ome_fname_base % _
            print('_',_, os.path.join(source_folder, ome_fname))

            if False:
                confocal.export_sr_ome(target_dir, source_folder, ome_fname,do_parallel=True, mock=False,
                                       n_channels=n_channels, keep3=keep3)  # 0, 1, 3
                # 1: 0 2 3
                # 2: 0 2 3

            globular_full = os.path.join(base_dir, specific_ds,
                                         'Intensity.png.export/Intensity*.png')  # 'AFM Test 100x/AFM Oversampling - Deconvolution/Intensity.png.export/*.png'
            print(len(glob.glob(globular_full)))
            destdir = os.path.join(base_dir, specific_ds)
            os.makedirs(os.path.join(destdir, 'objects'), exist_ok=True)
            desc = 'parent_%d' % _

            if True:
                confocal.max_projection(globular_full, do_plot=plot_max, do_save=True, destdir=destdir,
                                        filedesc=desc, as_colors=[0, 1, 2])
                print('Cell %d' % _)
                if plot_max: plt.show()

            objects_folder_base = source_folder  # os.path.join(source_folder, 'objects')
            base_folder = source_folder

            o = dict(
                globular_full=target_dir + '*.png',
                # reuse segmentation
                globular_nucleus=None, # os.path.join(objects_folder_base, 'objects/Nucleus*.png'),
                globular_cyto=None, # os.path.join(objects_folder_base, 'objects/Cytoplasm*.png'),
                source_folder=base_folder,  # diagnostics folder will be created here

                nuc_color=0,  # 2 0 2
                cyto_color=1,
                other_color=2,
            )

            cs1 = confocal.gen_load(o, use_masks=use_masks)

            if True:
                confocal.export_set_for_annotation(cs1, 'cyto_linear3D', use_log=False, do_show=False, do_parallel=True)
                confocal.export_set_for_annotation(cs1, 'other_linear3D', use_log=False, do_show=False, do_parallel=True)
                confocal.export_set_for_annotation(cs1, 'nucleus_linear3D', use_log=False, do_show=False, do_parallel=True)
    #         break

if __name__ == '__main__':
    main()
