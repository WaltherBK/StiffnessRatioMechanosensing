import argparse
import confocal
import contour
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import numpy as np

def main(output_folder, annotation_folder, probe_info=None, npts=21, padding=20, mirror_x=0, mirror_y=0):

    tags = ['nucleus_linear3D', 'cyto_linear3D', 'other_linear3D']

    envelope_nucleus = contour.convert_annotated(annotation_folder, 'nucleus_linear3D')
    envelope_cyto = contour.convert_annotated(annotation_folder, 'cyto_linear3D')
    envelope_other = contour.convert_annotated(annotation_folder, 'other_linear3D')

    ee = deepcopy(envelope_cyto)
    ne = deepcopy(envelope_nucleus) if len(envelope_nucleus) else deepcopy(envelope_other)

    cyto_padded = contour.reparam_contour3(ee, ne, doPlot=False, padding=padding, npts=npts)
    nuc_padded = contour.reparam_contour3(ne, None, doPlot=False, padding=padding, npts=npts)

    ###################
    # Caps and overhang
    if False:
        for n in list(sorted(nuc_padded.keys())):
            if n not in envelope_cyto:
                cyto_padded[n] = deepcopy(nuc_padded[n])
                print('Adding cyto layer to match nuc', n)

    if False:
        min_ne = min(nuc_padded.keys())
        if min_ne - 1 not in cyto_padded:
            print('Adding bottom layer')
            cyto_padded[min_ne - 1] = deepcopy(nuc_padded[min_ne])

    cyto_padded = contour.reparam_contour3(cyto_padded, nuc_padded, doPlot=False, padding=padding, npts=npts)
    nuc_padded = contour.reparam_contour3(nuc_padded, None, doPlot=False, padding=padding, npts=npts)

    cap_size_top = 2
    cap_size_bot = 2
    if True:
        # build up cyto above last nucleus
        max_ne = max(nuc_padded.keys())
        while max_ne >= max(cyto_padded.keys())-cap_size_top:
            print('Adding top cap at: ', max(cyto_padded.keys())+1)
            cyto_padded[max(cyto_padded.keys())+1] = deepcopy(cyto_padded[max(cyto_padded.keys())])

        min_ne = min(nuc_padded.keys())
        print('min:', min_ne, min(cyto_padded.keys()))
        while min_ne <= min(cyto_padded.keys())+cap_size_bot:
            print('Adding bottom cap at: ', min(cyto_padded.keys())-1)
            cyto_padded[min(cyto_padded.keys())-1] = deepcopy(cyto_padded[min(cyto_padded.keys())])

    if False:
        max_ne = max(nuc_padded.keys())
        cap_size = 0.3
        if max_ne + 1 not in cyto_padded:
            print('Adding cap layer')
            cyto_padded[max_ne + 1] = deepcopy(cyto_padded[max_ne])
            if cap_size > 0:
                cyto_padded[max_ne + 1] = deepcopy(cyto_padded[max_ne])
            if cap_size > 0.1:
                cyto_padded[max_ne + 2] = deepcopy(cyto_padded[max_ne])
            if cap_size > 0.2:
                cyto_padded[max_ne + 3] = deepcopy(cyto_padded[max_ne])


        # maybe a while loop?
        cap_size = 0.3
        min_ne = min(cyto_padded.keys())
        if min_ne >= min(nuc_padded.keys()):
            print('Adding bottom cap layer')
            if cap_size > 0:
                cyto_padded[min_ne - 1] = deepcopy(cyto_padded[min_ne])
            if cap_size > 0.1:
                cyto_padded[min_ne - 2] = deepcopy(cyto_padded[min_ne])
            if cap_size > 0.2:
                cyto_padded[min_ne - 3] = deepcopy(cyto_padded[min_ne])

    #################
    # Countour Healing

    # Bad values, because they don't account for centering
    tp0 = contour.transform_probe(probe_info[0], mirror_x=mirror_x, mirror_y=mirror_y)
    tp1 = contour.transform_probe(probe_info[1], mirror_x=mirror_x, mirror_y=mirror_y)

    print('transformed:', tp0, tp1)
    cyto_contour, moved = confocal.heal_contour(cyto_padded, probe_marker=tp0)
    print('moved:', moved)
    # # print(res.keys())
    nucleus_contour, moved = confocal.heal_contour(nuc_padded, move_to_mean=moved, probe_marker=tp1)
    print('moved again:', moved)
    sf = contour.get_scaling_factor(open(os.path.join('datasets/SR_HUVEC1/HUVEC Cell 1/annotation', 'scaling.json')))

    # plot healed
    plt.plot(cyto_contour[min(cyto_contour.keys())][0],cyto_contour[min(cyto_contour.keys())][1])
    plt.plot(nucleus_contour[min(nucleus_contour.keys())][0],nucleus_contour[min(nucleus_contour.keys())][1])
    plt.title('after healed')

    print('moved is:', moved)
    new_peak0 = [(tp0[0]-moved[0])*(0.03237/sf)**0,  (tp0[1]-moved[1])*(0.03237/sf)**0]
    new_peak1 = [(tp1[0]-moved[0])*(0.03237/sf)**0,  (tp1[1]-moved[1])*(0.03237/sf)**0]

    print('new_peaks:', new_peak0, new_peak1)
    plt.scatter(new_peak0[0],new_peak0[1], marker='o', color='purple')
    plt.scatter(new_peak1[0],new_peak1[1], marker='o', color='green')


    plt.show()


    os.makedirs(output_folder, exist_ok=True)

    # Export
    cyto_meta = confocal.export_contours(cyto_contour, ncvar='NCY', filen='C_Cyto_Z', scaling=(0.03237/sf, 0.03237/sf, 0.1),
                export_dir=output_folder, color_range='green', move_to_mean=None, skip_z=3, linestyle='dotted')
    nuc_meta = confocal.export_contours(nucleus_contour, ncvar='NC', filen='C_Nuc_Z', scaling=(0.03237/sf, 0.03237/sf, 0.1),
                export_dir=output_folder, color_range='red', move_to_mean=cyto_meta['move_to_mean'], skip_z=3, linestyle='dashed', linewidth=1)
    print('after export plot show', [tp0, tp1])


    plt.scatter(new_peak0[0],new_peak0[1], marker='o', color='purple')
    plt.scatter(new_peak1[0],new_peak1[1], marker='o', color='green')

    plt.show()

    print('move_to_mean:', cyto_meta['move_to_mean'], 'moved', moved)
    contour.export_probes(probes=[tp0, tp1], export_dir=output_folder, moved=moved, move_to_mean=cyto_meta['move_to_mean'], scaling=0.03237/sf)

    # move_to_mean: x_out = (x - xm) * scaling[0]



    meta_fname = os.path.join(output_folder,'file.txt')
    confocal.export_contours_meta('AFM-C HUVEC 20200626 Group #1, Cell #2', meta_fname, cyto_meta, nuc_meta)

    # Verification image
    nuc_re = contour.load_exported(os.path.join(output_folder, 'C_Nuc_Z01.txt'))
    cyto_re = contour.load_exported(os.path.join(output_folder, 'C_Cyto_Z01.txt'))
    p_c = [float(_.strip()) for _ in
           open(os.path.join(output_folder, 'probe_cyto.txt')).readlines()]
    p_n = [float(_.strip()) for _ in
           open(os.path.join(output_folder, 'probe_nuclear.txt')).readlines()]

    plt.plot(cyto_re.T[0], cyto_re.T[1], color='green', linestyle=':')
    plt.scatter(p_c[0], p_c[1], marker='o', color='green')
    plt.plot(nuc_re.T[0], nuc_re.T[1], linestyle=':', color='red')
    plt.scatter(p_n[0], p_n[1], marker='x', color='red')
    plt.title('Final scale, verification plot')
    plt.xlabel('(um)')
    plt.ylabel('(um)')

    plt.savefig(os.path.join(output_folder,'probe_positions.png'))
    plt.show()


if __name__ == '__main__':

    all_tips = np.load(os.path.join('datasets/SR_HUVEC1/Overlay Images HUVEC/', 'centered_all.npy'), allow_pickle=True)
    all_xy_mirror = [[None, 1, 0, 0, 0, 0, 0, 0], [None, 1, 0, 0, 0, 0, 0, 0]]


    for _ in range(1,7+1): # [1,]: # bad: 4 (no files), 6, 7 (no files)

        mirror_x = all_xy_mirror[0][_]
        mirror_y = all_xy_mirror[1][_]

        print('Starting to export %d' % _)

        print('tips: ', [all_tips[0][_][:2], all_tips[1][_][:2]])
        main(output_folder="export_contours/test_HUVEC1_%d" % _,
             annotation_folder='datasets/SR_HUVEC1/HUVEC Cell %d/annotation' % _,
             probe_info=[all_tips[0][_][:2], all_tips[1][_][:2]],
             mirror_x=mirror_x,
             mirror_y=mirror_y,
             # nuc_option='nucleus' if _ != 7 else 'other',
        )
