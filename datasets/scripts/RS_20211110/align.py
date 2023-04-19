# align.py
# aligns the pointers for a set of cells

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mplcursors

all_cells = range(1,7+1)
all_cells = [2,3,4,5,6,7]
all_cells = [4]

def main():

    base_folder = 'datasets/RS/'
    coord_folder = os.path.join(base_folder, 'RS AFM Coordinate Overlays')

    # fname_overlay_cyto = os.path.join(coord_folder,'HGPS Cell 1 Cyt.png')
    # fname_overlay_nuc = os.path.join(coord_folder,'HGPS Cell 1 Nuc.png')
    # fname_tip = os.path.join(coord_folder,'tip.png')

    view_types = ['Cyt', 'Nuc']
    view_type = 0
    pointers = []

    for cell_n in all_cells:

        fname_overlay_view = os.path.join(coord_folder, 'RS Cell %d %s.png' % (cell_n, view_types[view_type]))

        # print('overlay_view', fname_overlay_view)

        # fname_tip = os.path.join(coord_folder, 'tip.png')

        if not os.path.isfile(fname_overlay_view): print('DNE:', fname_overlay_view)
        img1_orig = cv2.imread(fname_overlay_view)

        # print('path:', fname_overlay_view)

        img1 = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2GRAY)

        # img1 = cv2.Canny(img1,10,20,100)
        # img2 = cv2.Canny(img2,50,100)

        fig = plt.figure(figsize=(16, 8))

        centered_all = [
            [
                (None, None, None),
                [1160.00000001, 1572.51428572,   60.],
                [1563.64999999, 1272.84285714, 90.],
                [1058.47142857,  800.7       , 60.],
                [1300.93961039, 1010.43571429, 100.],
                [1584.7538961, 1329.56428571, 60.],
                [766.43181818, 1479.23571429,   80.],
                [1054.75714286, 1623.47142858, 100.],

            ],

            [(None, None, None),
             [951.79220779, 1065.17857142, 90.],
             [769.66103896, 755.82142857, 90.],
             [944.96229127, 1519.84489796, 90.],
             [509.33961038, 1204.15, 90.],
             [1104.62207792, 1166.12857143, 125.],
             [1113.33181818,  733.45714285, 80.],
             [1327.81428571, 980.37142858, 90.]]
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
        if len(pointers): print('most recent pointer:', pointers[-1])

    print(np.array2string(np.array(pointers), separator=', '))


if __name__ == '__main__':
    main()
