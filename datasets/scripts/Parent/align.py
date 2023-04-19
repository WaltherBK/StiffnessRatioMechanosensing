# align.py
# aligns the pointers for a set of cells

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mplcursors

all_cells = range(1,7+1)
# all_cells = [7,]

def main():

    base_folder = 'datasets/Parent/'
    coord_folder = os.path.join(base_folder, 'Parent AFM Coordinate Overlays')

    # fname_overlay_cyto = os.path.join(coord_folder,'HGPS Cell 1 Cyt.png')
    # fname_overlay_nuc = os.path.join(coord_folder,'HGPS Cell 1 Nuc.png')
    # fname_tip = os.path.join(coord_folder,'tip.png')

    view_types = ['Cyt', 'Nuc']
    view_type = 0
    pointers = []

    for cell_n in all_cells:

        fname_overlay_view = os.path.join(coord_folder, 'Parent Cell %d %s.png' % (cell_n, view_types[view_type]))

        # print('overlay_view', fname_overlay_view)

        # fname_tip = os.path.join(coord_folder, 'tip.png')

        img1_orig = cv2.imread(fname_overlay_view)

        # print('path:', fname_overlay_view)

        img1 = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2GRAY)

        # img1 = cv2.Canny(img1,10,20,100)
        # img2 = cv2.Canny(img2,50,100)

        fig = plt.figure(figsize=(16, 8))

        centered_all = [
            [
                (None, None, None),
                [936.26818182, 1265.23571429, 110.],
                [598.65389611, 1047.5, 90.],
                [1334.2538961, 1230.02857143, 130.],
                [1508.47532467, 1010.54285714, 100.],
                [1685.72532467, 1008.15714286, 100.],
                [830.30389611, 455.83571429, 90.],
                [1597.86103896, 961.35714286, 110.]

            ],

            [(None, None, None),
             [541.02142858, 766.96428571, 110.],
             [1113.67142857, 708.77142857, 110.],
             [798.15714286, 1179.8, 110.],
             [743.58961039, 1025.11428572, 90.],
             [804.96103896, 1086.84285714, 120.],
             [1012.07532467, 1363.71428571, 80.],
             [1009.98571429, 1214.92857143, 150.],]
        ]

        np.save(os.path.join(coord_folder, 'centered_all.npy'), centered_all)

        # print('Selecting cell_n:', view_type, cell_n, centered_all[view_type][cell_n])
        centered = centered_all[view_type][cell_n]

        t = (centered[0] - 300, centered[0] + 300, centered[1] - 300, centered[1] + 300)
        # t = (400,1100,1600, 1100)
        # t = (700,850, 1300, 1200)

        # plt.xlim(t[0],t[1])
        # plt.ylim(t[2],t[3])
        # plt.scatter(centered[0], centered[1], marker='x', color='red')


        if True:
            thresh_val = centered[2]

            # top left
            fig, ax = plt.subplots(2, 2, figsize=(8, 8))
            ax[0][0].imshow(img1_orig, origin='lower')
            ax[0][0].set_title('Overlay orig')
            ax[0][0].scatter(centered[0], centered[1], marker='x', color='red')

            # top right
            ax[0][1].imshow(img1_orig, origin='lower')
            ax[0][1].set_xlim(centered[0] - 50, centered[0] + 50)
            ax[0][1].set_ylim(centered[1] - 50, centered[1] + 50)
            ax[0][1].scatter(centered[0], centered[1], marker='x', color='red')

            ax[0][1].set_title('Overlay zoom (%0.1f, %0.1f)' % (centered[0], centered[1]))

            # bottom left

            threshfactor = 2.0  # 1.75 # 1.0
            channel_finder = 1  # 0

            img1 = img1_orig[:, :, channel_finder]  # ::2
            #  ['BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV'][0]
            img1 = cv2.threshold(img1, thresh_val / threshfactor, 256, cv2.THRESH_BINARY)[1]
            img1 = cv2.Canny(img1, 10, 20)  # 10, 20

            ax[1][0].scatter(centered[0], centered[1], marker='x', color='red')
            ax[1][0].imshow(img1, origin='lower')

            ax[1][0].set_title('Original threshold + edges')

            # bottom right
            ax[1][1].imshow(img1)
            ax[1][1].set_xlim(centered[0] - 50, centered[0] + 50)
            ax[1][1].set_ylim(centered[1] - 50, centered[1] + 50)
            ax[1][1].scatter(centered[0], centered[1], marker='x', color='red')

            ax[1][1].set_title('Zoom threshold + edges')

            # overlay to top right
            ax[0][1].imshow(img1, alpha=0.1)

            fig.savefig(os.path.join(coord_folder, 'Cell %d %s.composite.png' % (cell_n, view_types[view_type])),
                        bbox_inches='tight')

            # plt.ion()
            cursor = mplcursors.cursor()
            cursor.connect("add", lambda sel: pointers.append(np.concatenate([sel.target, [centered[2],]])))

            # plt.clf()

            # plt.imshow(img1_orig, origin='lower')
            plt.show()

            # plt.draw()
            # plt.pause(0.001)
            # input("Press [enter] to continue.")
        print('most recent pointer:', pointers[-1])

    print(np.array2string(np.array(pointers), separator=', '))


if __name__ == '__main__':
    main()
