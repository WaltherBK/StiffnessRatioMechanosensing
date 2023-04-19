import argparse
import confocal
import contour
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import argparse

from contour import smoothed, bulge, pplinecuts, re_xy

parser = argparse.ArgumentParser(description='Build contours')
parser.add_argument('pfile', type=str,
                    help='json parameter file')

args = parser.parse_args()
argsd = vars(args)
print('argsd:', argsd)
json_opts = json.load(open(argsd['pfile'],'r'))
opts = json_opts['build']

EXPORT_CONTOURS_TO = opts['export_to'] # "export_contours/contours_hgps_P%02d_%d_redo"
ANNOTATION_FOLDER = opts['annotation_folder'] # 'HGPS %d/annotation'
TITLE = opts['title'] # 'HGPS_%d'
COMMENT = opts['comment'] # "AFM-C HGPS Cell #%d"
SCALING_JSON = os.path.join(opts['scaling_json_dir'], opts['scaling_json_fname'])
Z_SCALE = opts['z_scale'] # 0.1
BASE_FOLDER = opts['base_folder'] # 'datasets/SR_HGPS1/'
COORD_FOLDER = os.path.join(BASE_FOLDER, opts['coord_folder'] )
SMOOTH_W = opts['smooth_w'] # 3
HOLD_EDGE = opts['hold_edge'] # True

DO_PLOT = False

def main(output_folder, annotation_folder, probe_info=None, npts=21, padding=20, mirror_x=0, mirror_y=0,
         title='title', comment='comment', doPlot=False, z_scale=None):

    tags = ['nucleus_linear3D', 'cyto_linear3D', 'other_linear3D']

    # Load annotations
    envelope_nucleus = contour.convert_annotated(annotation_folder, 'nucleus_linear3D')
    envelope_cyto = contour.convert_annotated(annotation_folder, 'cyto_linear3D')
    envelope_other = contour.convert_annotated(annotation_folder, 'other_linear3D')

    ee = deepcopy(envelope_cyto)
    ne = deepcopy(envelope_nucleus) if len(envelope_nucleus) else deepcopy(envelope_other)

    # Find center of mass at top:
    last_level = list(sorted(ne.keys()))[-1]
    print('Last level polygon:', ne[last_level])
    rows = ne[last_level][:2].T.astype(np.float32)
    M = cv2.moments(rows)
    recentering = M["m10"] / M["m00"], M["m01"] / M["m00"]
    print('Recentered:', recentering)
    # raise Exception('stop')

    # print(ee)
    s = np.array(json.load(open(SCALING_JSON))['shapes'][0]['points']).T

    if doPlot:
        ul = [max(ee.keys()), max(ne.keys())]
        ll = [min(ee.keys()), min(ne.keys())]

        plt.plot(ee[ll[0]][0], ee[ll[0]][1], color='green') # cyto lower
        plt.plot(ne[ll[1]][0], ne[ll[1]][1],color='orange') # nucleus lower
        plt.plot(ee[ul[0]][0], ee[ul[0]][1],color='green',linestyle=':') # cyto upper
        plt.plot(ne[ul[1]][0], ne[ul[1]][1],color='orange',linestyle=':') # nucleus upper

        # Should not show probes; contours traced on axes plots, probes were not.
        # plt.scatter(*probe_info[0])
        # plt.scatter(*probe_info[1])

        print('probe_info[0]', probe_info[0])
        print('probe_info[0]', probe_info[0])
        print('s:', s)
        tp0 = contour.probe_to_labelme(probe_info[0],s)
        tp1 = contour.probe_to_labelme(probe_info[1],s)
        print('tp0', tp0)
        print('tp1', tp1)

        plt.scatter(*tp0, marker='x')
        plt.scatter(*tp1)

        plt.title('Raw top/bottom layers')
        plt.show()

    # Calls edit_level on each level, which expands out, and does some other conditioning.
    # nuclear envelope included so that cytoplasm is always outside
    cyto_padded = contour.reparam_contour4(ee, ne, doPlot=False, padding=padding, npts=npts, recentering=recentering)
    # no such req for nucleus itself
    nuc_padded = contour.reparam_contour4(ne, None, doPlot=False, padding=padding, npts=npts, recentering=recentering)
    #
    # for PL in range(20):  # PL are the thetas
    #     all_levs = list(sorted(set(nuc_padded.keys()).union(cyto_padded.keys())))
    #     lp = np.linspace(all_levs[0], all_levs[-1], 10)
    #
    #     levs = sorted(nuc_padded.keys())
    #     x = np.array([nuc_padded[lev][0][PL] - recentering[0] for lev in levs])
    #     y = np.array([nuc_padded[lev][1][PL] - recentering[1] for lev in levs])
    #     #     plt.plot(x,y)
    #
    #     lpt = [_ for _ in lp if levs[0] <= _ <= levs[-1]]  # interpolate only at the points inside valid region
    #     levsp, rpp = make10_nondec(np.sqrt(x ** 2 + y ** 2), levs, lp=lpt, do_smooth=True)
    #     levspg, rppg = levsp, rpp  # save as guide
    #
    #     plt.scatter(rpp, levsp)
    #     plt.plot(np.sqrt(x ** 2 + y ** 2), levs, linestyle=':')
    #     print('nuc vals:', rpp, levsp)
    # #     plt.xscale('log')
    #

    # Have to add cap before interpolating, otherwise we may break constant number of layers.
    cap_size_top = opts['cap_size_top'] # 2
    cap_size_bot = opts['cap_size_bot'] # 2

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

    aux_out = os.path.join(output_folder, 'aux')
    os.makedirs(aux_out,exist_ok=True)
    os.makedirs(os.path.join(output_folder,'imgs'),exist_ok=True)


    if doPlot:
        ncxy = cyto_padded
        nnxy = nuc_padded
        ul = [max(ncxy.keys()), max(nnxy.keys())]
        ll = [min(ncxy.keys()), min(nnxy.keys())]

        plt.plot(ncxy[ll[0]][0], ncxy[ll[0]][1], color='green') # cyto lower
        plt.plot(nnxy[ll[1]][0], nnxy[ll[1]][1],color='orange') # nucleus lower
        plt.plot(ncxy[ul[0]][0], ncxy[ul[0]][1],color='green',linestyle=':') # cyto upper
        plt.plot(nnxy[ul[1]][0], nnxy[ul[1]][1],color='orange',linestyle=':') # nucleus upper
        plt.title('Padded layers')

        tp0 = contour.probe_to_labelme(probe_info[0],s)
        tp1 = contour.probe_to_labelme(probe_info[1],s)
        print('tp0', tp0)
        print('tp1', tp1)

        plt.scatter(*tp0, marker='x')
        plt.scatter(*tp1)

        plt.show()

    nn, nc = pplinecuts(nuc_padded, cyto_padded, recentering=recentering, doPlot=False, savePlot=aux_out, bb=1,
            mva=SMOOTH_W, hold_edge=HOLD_EDGE, nangles=opts['interpolation_pts'])

    nnxy = re_xy(nn, npts=opts['interpolation_pts'])
    ncxy = re_xy(nc, npts=opts['interpolation_pts'])

    if doPlot:
        ul = [max(ncxy.keys()), max(nnxy.keys())]
        ll = [min(ncxy.keys()), min(nnxy.keys())]

        plt.plot(ncxy[ll[0]][0], ncxy[ll[0]][1], color='green') # cyto lower
        plt.plot(nnxy[ll[1]][0], nnxy[ll[1]][1],color='orange') # nucleus lower
        plt.plot(ncxy[ul[0]][0], ncxy[ul[0]][1],color='green',linestyle=':') # cyto upper
        plt.plot(nnxy[ul[1]][0], nnxy[ul[1]][1],color='orange',linestyle=':') # nucleus upper
        plt.title('Linecutted layers')

        tp0 = contour.probe_to_labelme(probe_info[0],s, recentering=recentering)
        tp1 = contour.probe_to_labelme(probe_info[1],s, recentering=recentering)
        print('recentering:', recentering)

        plt.scatter(*tp0, marker='x')
        plt.scatter(*tp1)



        plt.show()

    print('nnxy:', nnxy)
    nucleus_contour = nnxy
    cyto_contour = ncxy

    moved = recentering

    # pts_array = np.array([cyto_padded[lev] for lev in sorted(cyto_padded.keys())]).transpose([2,1,0])
    # for zz in range(len(pts_array)):
    #     plt.plot(pts_array[zz][0],pts_array[zz][1])
    # plt.show()

    # raise Exception('stop')


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

    # cyto_padded = contour.reparam_contour4(cyto_padded, nuc_padded, doPlot=False, padding=padding, npts=npts, recentering=recentering)
    # nuc_padded = contour.reparam_contour4(nuc_padded, None, doPlot=False, padding=padding, npts=npts, recentering=recentering)




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
    tp0 = contour.probe_to_labelme(probe_info[0], s)
    tp1 = contour.probe_to_labelme(probe_info[1], s)

    # print('transformed:', tp0, tp1)
    # cyto_contour, moved = confocal.heal_polar(cyto_padded, probe_marker=tp0) # don't need to heal anymore
    # print('moved:', moved)
    # # print(res.keys())
    # nucleus_contour, moved = confocal.heal_polar(nuc_padded, move_to_mean=moved, probe_marker=tp1)
    # print('moved again:', moved)

    sf = contour.get_scaling_factor(open(SCALING_JSON))
    rf = opts['resolution_factor'] # 0.03237

    # plot healed
    if doPlot:
        plt.plot(cyto_contour[min(cyto_contour.keys())][0],cyto_contour[min(cyto_contour.keys())][1])
        plt.plot(nucleus_contour[min(nucleus_contour.keys())][0],nucleus_contour[min(nucleus_contour.keys())][1])
        plt.title('after healed')
        plt.savefig(os.path.join(output_folder,'imgs/healed_plot.png')) # XYZ

    print('moved is:', moved)
    new_peak0 = [(tp0[0]-moved[0])*(rf/sf)**0,  (tp0[1]-moved[1])*(rf/sf)**0]
    new_peak1 = [(tp1[0]-moved[0])*(rf/sf)**0,  (tp1[1]-moved[1])*(rf/sf)**0]

    print('new_peaks:', new_peak0, new_peak1)
    if doPlot:
        plt.scatter(new_peak0[0],new_peak0[1], marker='o', color='purple')
        plt.scatter(new_peak1[0],new_peak1[1], marker='o', color='green')
        plt.savefig(os.path.join(output_folder,'imgs/new_peaks.png')) # XYZ
        plt.show()


    os.makedirs(output_folder, exist_ok=True)

    # Export
    cyto_meta = confocal.export_contours(cyto_contour, ncvar='NCY', filen='C_Cyto_Z', scaling=(rf/sf, rf/sf, z_scale),
                export_dir=output_folder, color_range='green', move_to_mean=None, skip_z=1, linestyle='dotted',
                filename_type=('CYTO', 'Cyto', title, comment))
    nuc_meta = confocal.export_contours(nucleus_contour, ncvar='NC', filen='C_Nuc_Z', scaling=(rf/sf, rf/sf, z_scale),
                export_dir=output_folder, color_range='red', move_to_mean=cyto_meta['move_to_mean'], skip_z=1, linestyle='dashed', linewidth=1,
                filename_type=('NUC', 'Nuc', title, comment))
    print('after export plot show', [tp0, tp1])

    if doPlot:

        plt.scatter(new_peak0[0],new_peak0[1], marker='o', color='purple')
        plt.scatter(new_peak1[0],new_peak1[1], marker='o', color='green')
        plt.savefig(os.path.join(output_folder,'imgs/scatter_plot.png')) # XYZ

        plt.show()

    print('move_to_mean:', cyto_meta['move_to_mean'], 'moved', moved)
    # contour.export_probes(probes=[tp0, tp1], export_dir=output_folder, moved=moved, move_to_mean=cyto_meta['move_to_mean'], scaling=rf/sf)

    # move_to_mean: x_out = (x - xm) * scaling[0]

    if moved is not None:
        tp0 -= moved
        tp1 -= moved

    scaling = rf/sf
    tp0 *= scaling
    tp1 *= scaling

    meta_fname = os.path.join(output_folder,'CONT_DATA_%s.txt' % title)
    # confocal.export_contours_meta('AFM-C HUVEC 20200626 Group #1, Cell #2', meta_fname, cyto_meta, nuc_meta)
    data = {
        'fname': 'CONT_DATA_%s.txt' % title,
        'comment': comment,
        'DZ': opts['DZ'],
        'xn':tp0[0],
        'yn':tp0[1],
        'xc':tp1[0],
        'yc':tp1[1],

    }
    confocal.export_contours_newmeta(comment, meta_fname, cyto_meta, nuc_meta, data)

    # Verification image
    nuc_re = contour.load_exported(os.path.join(output_folder, 'C_Nuc_Z01.txt'))
    cyto_re = contour.load_exported(os.path.join(output_folder, 'C_Cyto_Z01.txt'))
    p_c = tp0
    p_n = tp1

    # p_c = [float(_.strip()) for _ in
    #        open(os.path.join(output_folder, 'probe_cyto.txt')).readlines()]
    # p_n = [float(_.strip()) for _ in
    #        open(os.path.join(output_folder, 'probe_nuclear.txt')).readlines()]

    plt.clf()
    plt.plot(cyto_re.T[0], cyto_re.T[1], color='green', linestyle=':')
    plt.scatter(p_c[0], p_c[1], marker='o', color='green')
    plt.plot(nuc_re.T[0], nuc_re.T[1], linestyle=':', color='red')
    plt.scatter(p_n[0], p_n[1], marker='x', color='red')
    plt.title('Final scale, verification plot')
    plt.xlabel('(um)')
    plt.ylabel('(um)')
    print('Probes:', p_n, p_c)

    plt.savefig(os.path.join(output_folder,'probe_positions.png'))

    if doPlot or False:
        plt.show()

    plt.clf()


if __name__ == '__main__':

    base_folder = BASE_FOLDER
    coord_folder = COORD_FOLDER

    centerings_fname = os.path.join(coord_folder, 'centered_all.npy')
    all_tips = np.load(centerings_fname, allow_pickle=True)
    print('Loaded tips:', centerings_fname, all_tips)

    all_xy_mirror = opts['all_xy_mirror'] # [[None, 0, 0, 0, 0, 0, 0, 0, 0], [None, 1, 1, 0, 0, 0, 0, 0, 0]]

    all_padding = opts['all_padding'] # [0,10,20,30,40,80] # 10, 20, 30, 40, 80
    for _ in opts['all_cells']: # [1,2,4,5,6,7,8]: #

        for pn in range(len(all_padding)):

            # if pn < 5: continue
            padding = all_padding[pn]

            mirror_x = all_xy_mirror[0][_]
            mirror_y = all_xy_mirror[1][_]

            print('Starting to export %d' % _)

            z_scale = Z_SCALE[_] if (type(Z_SCALE) == list) else Z_SCALE

            print('tips: ', [all_tips[0][_][:2], all_tips[1][_][:2]])
            main(output_folder=EXPORT_CONTOURS_TO % (padding, _),
                 annotation_folder=os.path.join(base_folder,ANNOTATION_FOLDER % _),
                 probe_info=[all_tips[0][_][:2], all_tips[1][_][:2]],
                 mirror_x=mirror_x,
                 mirror_y=mirror_y,
                 title=TITLE % _,
                 comment=COMMENT % _,
                 padding=padding,
                 doPlot=DO_PLOT,
                 npts=opts['interpolation_pts'],
                 z_scale=z_scale,
                 # nuc_option='nucleus' if _ != 7 else 'other',
            )
