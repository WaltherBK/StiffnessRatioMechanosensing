import glob, os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import imageio
import re
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import interp2d,RegularGridInterpolator, interp1d
from random import random
from tqdm import tqdm
import json
from copy import deepcopy
import contour

from itertools import accumulate
from skimage.measure import find_contours
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter

from skimage import io
import imageio

import itertools
import cv2

import multiprocessing as mp

import argparse
import PIL
import matplotlib
import imageio


def rewrite_file(args):
    target_dir, fname_stub, i, fname = args

    data = io.imread(fname)
    print(fname_stub + '%03d' % i, data.shape)

    # io.imsave(os.path.join(target_dir, fname_stub+'%03d.png' % i),data)
    out_path = os.path.join(target_dir, fname_stub + '%03d.png' % i)
    imageio.imwrite(out_path, data, format='PNG-FI')
    return out_path

def testf(args):
    a,b = args
    print(a,b)
    return b

def zipdump_png(args):
    target_dir, fname_stubs, i, data, keep3 = args

    # data =

def dump_png(args):
    target_dir, fname_stub, i, data, keep3 = args

    data = np.transpose(data,axes=[1,2,0])

    data = data[:,:,keep3]
    print(fname_stub + '%03d' % i, data.shape)

    # io.imsave(os.path.join(target_dir, fname_stub+'%03d.png' % i),data)
    out_path = os.path.join(target_dir, fname_stub + '%03d.png' % i)
    imageio.imwrite(out_path, data, format='PNG-FI')
    return out_path


def export_sr_tifs(target_dir, source_folder, fname_stub='Intensity_Z', as_colors=None):
    fnames_tif = list(sorted(glob.glob(os.path.join(source_folder, '*.tif'))))
    print(len(fnames_tif))

    os.makedirs(os.path.join(target_dir, 'Intensity.png.export/'), exist_ok=True)



    with mp.Pool() as p:

        arg_list = [(target_dir, fname_stub, i, fname) for i, fname in enumerate(fnames_tif)]
        #         print()

        # res = p.map(testf, arg_list)
        res = p.map(rewrite_file, arg_list)

    print('res',res)
        # rewrite_file(target_dir, fname_stub, i, fname)


# Special export of OME file into images
def export_sr_ome(target_dir, source_folder, fname, fname_stub='Intensity_Z', as_colors=None, n_channels=3,
                  keep3=[0,1,2], do_parallel=True, mock=False, ome_splitted=False):
    from PIL import Image, ImageSequence

    # fnames_tif = list(sorted(glob.glob(os.path.join(source_folder, fname))))

    os.makedirs(target_dir, exist_ok=True)

    if not ome_splitted: # typical case, interleaved ome.
        fname_ome = os.path.join(source_folder, fname)

        def divide_chunks(l, n):
            # looping till length l
            for i in range(0, len(l), n):
                yield l[i:i + n]

        try:
            im = Image.open(fname_ome)
        except:
            if os.path.isfile(fname_ome):
                raise Exception('Bad file:' + fname_ome)
            else:
                raise Exception('File DNE:' + fname_ome)

        iterated = [np.array(_) for _ in ImageSequence.Iterator(im)]

        arg_list = [(target_dir, fname_stub, i, data, keep3) for i, data in enumerate(divide_chunks(iterated,n_channels))]

    else: # split files, so fname_ome is a list
        fnames_ome = [os.path.join(source_folder, _) for _ in fname]
        print('fnames: ', fnames_ome)
        ims = [Image.open(_) for _ in fnames_ome]
        ims = np.array([[np.array(_) for _ in ImageSequence.Iterator(im)] for im in ims])
        print('Loaded.')


        arg_list = [(target_dir, fname_stub, i, data, keep3) for i, data in
                    enumerate(ims.transpose([1,0,2,3]))]

    if do_parallel and not mock:
        with mp.Pool() as p:
        #
            res = p.map(dump_png, arg_list)
    else:
        if not mock:
            res = [dump_png(ii) for ii in arg_list]
        else:
            for ii in arg_list:
                print('Dumping:', ii)


def export_tif_old(target_dir, fname_tif):
    from PIL import Image, ImageSequence

    # target_dir = os.path.join('datasets/AFM Test 100x/AFM Test 2 - LMNA(g) Actin(r).export/', 'Intensity.png.export')
    # im = Image.open(os.path.join('datasets/AFM Test 100x/AFM Test 2 - LMNA(g) Actin(r).export/', 'Intensity.tif.export',
    #                              "Intensity_45Z_CH.ome.tif"))

    im = Image.open(fname_tif)

    os.makedirs(target_dir, exist_ok=True)
    for i, page in tqdm(enumerate(ImageSequence.Iterator(im))):
        page.save(os.path.join(target_dir, "Intensity_Z%03d.png" % i))
    #     break
    print('done')

def export_tif(target_dir, fname_tif, fname_stub='Intensity_Z', as_colors=None):

    im = io.imread(fname_tif)
    print('Data shape:', im.shape)

    n_channels = len(im)
    if as_colors is None:
        as_colors = list(range(3))
        as_colors[1] = 1 if n_channels > 1 else 0
        as_colors[2] = 2 if n_channels > 2 else 0
    print('Using channel colors:', as_colors)

    im = np.transpose(im, axes=[1,2,3,0])

    for i in tqdm(range(len(im))):
        data = np.array(im[i][:,:,as_colors],dtype=np.uint16)*16
        # io.imsave(os.path.join(target_dir, fname_stub+'%03d.png' % i),data)
        imageio.imwrite(os.path.join(target_dir, fname_stub+'%03d.png' % i),data, format='PNG-FI')

    # target_dir = os.path.join('datasets/AFM Test 100x/AFM Test 2 - LMNA(g) Actin(r).export/', 'Intensity.png.export')
    # im = Image.open(os.path.join('datasets/AFM Test 100x/AFM Test 2 - LMNA(g) Actin(r).export/', 'Intensity.tif.export',
    #                              "Intensity_45Z_CH.ome.tif"))

def heal_contour(contours, move_to_mean=None, do_connect=False, doPlot=True, probe_marker=None):
    if move_to_mean is None:
        print('Healing: moving to unprovided mean')
        xm = np.mean(np.hstack([contours[_][0] for _ in contours]))
        ym = np.mean(np.hstack([contours[_][1] for _ in contours]))
    else:
        print('Healing: moving to given mean', move_to_mean)
        xm, ym = move_to_mean

    new_contours = {}
    for level, c in contours.items():

        # re-center contour

        c = np.array([[_ - xm for _ in c[0]], [_ - ym for _ in c[1]]])

        # Connect beginning to end


        # Verify direction of rotation; reverse if necessary

        dx = c[0][1:] - c[0][:-1]
        dy = c[1][1:] + c[1][:-1]
        area = np.dot(dx, dy) / 2.0  # for all points, the sum of: (x2 − x1)(y2 + y1)
        #         print('area:',area)
        if area > 0:
            # print('reversing')
            # c = np.array([c[0][::-1], c[1][::-1]])  # reverse
            c = np.array([list(reversed(c[0])), list(reversed(c[1]))])  # reverse

        # Identify farthest point in x, and rotate contour indices

        idx = np.argmax(c[0])
        c = np.roll(c, -idx, axis=1)
        idx = np.argmax(c[0])

        if do_connect:
            # print('Connecting!')
            c = np.hstack([c, [[c[0][0]], [c[1][0]]]])
            pass


        if doPlot:
            plt.plot(c[0], c[1])
            plt.scatter(c[0][idx], c[1][idx])
            if probe_marker is not None:
                plt.scatter(probe_marker[0]-xm, probe_marker[1]-ym, marker='x', color='red')

        new_contours[level] = deepcopy(c)

    if doPlot: plt.show()

    return new_contours, (xm, ym)

def heal_polar(contours, move_to_mean=None, do_connect=False, doPlot=True, probe_marker=None):
    if move_to_mean is None:
        print('Healing: moving to unprovided mean')
        xm = np.mean(np.hstack([contours[_][0] for _ in contours]))
        ym = np.mean(np.hstack([contours[_][1] for _ in contours]))
    else:
        print('Healing: moving to given mean', move_to_mean)
        xm, ym = move_to_mean

    new_contours = {}
    for level, c in contours.items():

        # re-center contour

        c = np.array([[_ - xm for _ in c[0]], [_ - ym for _ in c[1]]])

        new_contours[level] = deepcopy(c)

    if doPlot: plt.show()

    return new_contours, (xm, ym)

def reparam_contour(c, npts=21):
    #     print(c.keys())
    for level in c:
        #         print(c[level].shape)
        lx = c[level][0]
        ly = c[level][1]

        M = cv2.moments(c[level].T.astype(np.int32))
        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        dist = np.array(
            [0, ] + [np.sqrt((i1 - i0) ** 2 + (j1 - j0) ** 2) for i0, i1, j0, j1 in zip(lx[1:], lx, ly[1:], ly)])
        sumdist = np.array(list(itertools.accumulate(dist)))
        #         print('dist shape:', dist.shape, sumdist.shape, sumdist)

        fx = interp1d(sumdist, lx)
        fy = interp1d(sumdist, ly)

        gd = np.linspace(0, sumdist[-1], 101)
        gx, gy = fx(gd), fy(gd)
        gtheta = np.unwrap(np.arctan2((gy - cY), (gx - cX)))
        afterpt = np.argmax(gtheta > gtheta[0] + np.pi)

        #         print(gtheta)
        #         print('targets:')
        if False:
            plt.plot(gd, gtheta)
            #         plt.axhline(0)
            plt.show()

        if False:
            plt.scatter(gx, gy)

            #         plt.show()

            plt.plot(lx, ly)
            plt.scatter(cX, cY)

            print('Afterpt:', 0, afterpt, gtheta[0], gtheta[afterpt])
            plt.scatter(fx(gd[afterpt]), fy(gd[afterpt]))
            plt.scatter(lx[0], ly[0])

            plt.show()

        hd = gd[afterpt]  # halfway distance

        gd1 = np.linspace(0, hd, npts // 2 + 1)
        gd2 = np.linspace(hd, sumdist[-1], npts // 2)[1:]

        newx = np.hstack([fx(gd1), fx(gd2)])
        newy = np.hstack([fy(gd1), fy(gd2)])

        if False:
            plt.scatter(newx, newy)
            plt.title('broken up')
            plt.scatter(fx(gd[afterpt]), fy(gd[afterpt]))
            plt.scatter(cX, cY, marker='s')

            plt.scatter(lx[0], ly[0])
            plt.show()
        #         break
        #         print(np.shape(c[level]),np.shape(np.array([newx,newy])))
        c[level] = np.array([newx, newy])

    return c

# Export metadata for contours, as needed for Ansys processing
def export_contours_meta(title, fname, cyto_meta, nuc_meta):
    lines = []
    lines.append('!******************************************')
    lines.append('! %s' % title)
    lines.append('n_nuc=%d                        ! number of files of nuclear contours' % len(nuc_meta['files_list']))
    lines.append('n_cyt=%d                        ! number of files of cytoplasm contours' % len(cyto_meta['files_list']))
    lines.append('Z_cyt=%0.3f            ! (µm) Z value of the upper Z-slice (Cell Height)' % cyto_meta['z_max'])
    lines.append('Z_nuc=%0.3f           ! (µm) Z value of the upper Z-slice (Nucleus Height)' % nuc_meta['z_max'])
    lines.append('\n!')

    for f in cyto_meta['files_list']:
        lines.append(f)
    lines.append('!')
    for f in nuc_meta['files_list']:
        lines.append(f)

    lines.append('!******************************************')

    with open(fname, 'w') as fout:
        fout.write("\n".join(lines))

    print('Wrote meta (for %s) to %s' % (title, fname))

def export_contours_newmeta(title, fname, cyto_meta, nuc_meta, data):



    lines = []
    lines.append('!******************************************')
    lines.append('! File Name:  %s' % data['fname'])
    lines.append('! %s' % data['comment'])
    lines.append('!******************************************')
    lines.append('n_nuc=%d    ! numb of files of nucl cont' % len(nuc_meta['files_list']))
    lines.append('n_cyt=%d    ! numb of files of cyto cont' % len(cyto_meta['files_list']))
    lines.append('Z_cyt=%0.3f   ! (µm) Max Cyto Z-value (Cell Height)' % cyto_meta['z_max'])
    lines.append('Z_nuc=%0.3f   ! (µm) Max Nuc Z-value (Nuc Height)' % nuc_meta['z_max'])
    lines.append('Zmax=Z_cyt  ! (µm) Max height cyto')
    lines.append('DZ=%0.3f	     ! (µm) step Z-slices' % data['DZ'])
    lines.append('!---------------------------------------------------')
    lines.append('!Default the cent of Gravi of the higher nucl conto')
    lines.append('!----------------------------------------------------')
    lines.append('xn=%0.3f  ! (µm) X position AFM Probe nuc' % data['xn'])
    lines.append('yn=%0.3f ! (µm) Y position AFM Probe nuc' % data['yn'])
    lines.append('!')
    lines.append('xc=%0.3f !(µm) X position AFM Prob cyt' % data['xc'])
    lines.append('yc=%0.3f    !(µm) Y position AFM Probe cyt' % data['yc'])

    chaser = """
!
!--------
! Arrays
!--------
! associated to nucleus
!----------------------
*dim,NC,array,n_nuc
*dim,NChalf,array,n_nuc
*dim,NNN,array,n_nuc
*dim,NN2,array,n_nuc
*dim,NI,array,n_nuc
*dim,NP,array,n_nuc
*dim,NL,array,n_nuc
*dim,NR,array,n_nuc
!------------------------
! associated to cytoplasm
!------------------------
*dim,NCY,array,n_cyt
*dim,NCYhalf,array,n_cyt
*dim,NNNY,array,n_cyt
*dim,NN2Y,array,n_cyt
*dim,NIY,array,n_cyt
*dim,NPY,array,n_cyt
*dim,NLY,array,n_cyt
*dim,NRY,array,n_cyt
!******************************************
"""
    lines.append(chaser)

    with open(fname, 'w') as fout:
        fout.write("\n".join(lines))

    print('Wrote meta (for %s) to %s' % (title, fname))

def rem_zeroes(leveler):
    marked = []

    def find_zeroes(lev, contourn=-1):
        for n, (xx, yy) in enumerate(lev.T):
            dd = lev.T - [[xx, yy]]
            dists = np.sqrt(dd.T[0] ** 2 + dd.T[1] ** 2)
            dists = [d for nn, d in enumerate(dists) if nn != n]
            mind, argd = np.min(dists), np.argmin(dists)
            if mind < 1e-3:
                print('XX:', xx)
                newx = [_ for nn, _ in enumerate(lev[0]) if nn != n]
                newy = [_ for nn, _ in enumerate(lev[1]) if nn != n]
                # print('recursing level %d!' % contourn, argd, mind, dists)
                return find_zeroes(deepcopy(np.array([newx, newy])), contourn=contourn)
        return lev

    for _ in leveler:
        leveler[_] = find_zeroes(deepcopy(leveler[_]), contourn=_)

    return leveler

import matplotlib as mpl
def export_contours(contour_0, move_to_mean=(0,0,0), export_dir='export/', color_range='green', ncvar=None, filename_type=('','', ''),
                    filen='contour_export_', scaling=(1, 1, 1), half_idx=None, remove_zeroes=False, skip_z=1, linestyle='solid', linewidth=2):

    series, comment, title, desc = filename_type

    if move_to_mean is None:
        # cyto
        print('Moving to mean')
        xm = np.mean(np.hstack([contour_0[_][0] for _ in contour_0]))
        ym = np.mean(np.hstack([contour_0[_][1] for _ in contour_0]))
        zm = np.min(list(sorted(contour_0.keys()))[::skip_z])
    else:
        xm, ym, zm = move_to_mean

    levels = sorted(list(contour_0.keys()))[::skip_z]
    print('levels skip:', skip_z, levels)
    # levels.update(contour_1.keys())
    max_level = np.max(list(levels))
    min_level = np.min(list(levels))
    print('min/max level:', min_level, max_level)
    # level_subtraction = min_level - 1

    norm = mpl.colors.Normalize(vmin=min_level - 20, vmax=max_level + 20)

    if color_range=='green':
        cmap0 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)
        cmap0.set_array([])
    elif color_range=='blue':
        cmap0 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
        cmap0.set_array([])
    elif color_range=='red':
        cmap0 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)
        cmap0.set_array([])
    else:
        color_range = None
        print(print('plotting black contours because of invalid color_range'))

    if remove_zeroes:
        raise('Removing zeroes is deprecated')
        # print('removing zeroes')
        contour_0 = rem_zeroes(contour_0)

    saved_layer_files = []
    os.makedirs(export_dir, exist_ok=True)
    z_max = None
    for nz, sliceZ in enumerate(levels):

        nz+=1
        lines = []

        lines.append('!-----------------------')
        lines.append('! File Name : C_%s_Z%02d.txt' % (comment, nz))
        lines.append('!Contour #%02d at Z=%0.2f' % (nz, sliceZ))
        lines.append('!-----------------------')
        lines.append('%s(%d)=%d' % (ncvar, nz, len(contour_0[sliceZ][0])))
        if half_idx is not None: lines.append('%shalf(%d)=%d' % (ncvar, nz, half_idx))

        for x, y in zip(contour_0[sliceZ][0], contour_0[sliceZ][1]):
            x_out = (x - xm) * scaling[0]
            y_out = (y - ym) * scaling[1]
            z_out = (sliceZ- zm) * scaling[2]
            lines.append('k,,%0.3f,%0.3f,%0.3f' % (x_out, y_out, z_out))
            if z_max is None or z_out > z_max:
                # print('setting old z_max to z_out:', z_max, z_out)
                z_max = z_out

            c = cmap0.to_rgba(sliceZ) if color_range is not None else 'k'
            plt.plot(contour_0[sliceZ][0] - xm, contour_0[sliceZ][1] - ym, color=c, linestyle=linestyle, linewidth=linewidth)  # 'green')

        fname = filen + '%02d.txt' % (nz)
        saved_layer_files.append(fname)
        with open(os.path.join(export_dir, fname), 'w') as fout:
            fout.writelines('\n'.join(lines))

    plt.savefig(os.path.join(export_dir, filen + '.png'), dpi=300)

    # contents file
    with open(os.path.join(export_dir,'CONT_%s_%s.txt' % (series, title)),'w') as fout:

        contents = []
        contents.append('!******************************************')
        contents.append('! File name: CONT_%s_%s.txt' % (series,title))
        contents.append('! %s' % desc)
        contents.append('!******************************************')

        for nz, sliceZ in enumerate(levels):
            contents.append("/INPUT,'C_%s_Z%02d','txt'" % (series, nz+1))

        fout.write("\n".join(contents))


    return {'move_to_mean':(xm, ym, zm),'files_list':saved_layer_files, 'z_max':z_max}


def read_example_contour_file(fname='HUVEC-CTRL-C1-CONTOURS-Z00.txt'):
    excon = open(fname, 'r').readlines()
    import re
    regex = re.compile('#(\d)')
    compiled_data = {}
    current_curve = None
    for r in excon:
        #     print(r.strip())
        #     print()
        if r[0] != '!':
            data = r.strip().split(',')
            compiled_data[current_curve].append(data[2:])
        #         print(data)
        else:
            curve_test = regex.findall(r)
            if len(curve_test): current_curve = curve_test[0]
            if current_curve not in compiled_data: compiled_data[current_curve] = []

    compiled_data = {key: np.array(val, dtype=float).T for key, val in compiled_data.items()}

    compiled_data['1']
    plt.plot(compiled_data['1'][0], compiled_data['1'][1])
    plt.plot(compiled_data['2'][0] * 0.5 + 1.5, compiled_data['2'][1] * 0.5 + 1.5)
    plt.savefig('example_contours_given.png', dpi=300)

def max_projection_tif(fname, do_plot=False, do_save=None,destdir=None, filedesc=None,as_colors=None):

    print('Options:  fname, do_plot=False, do_save=None, destdir=None, filedesc=None,')
    import imageio
    im = io.imread(fname)
    print('Data shape:', im.shape)


    # im = np.transpose(im, axes=[1, 2, 3, 0])

    for cn,im_series_ch in enumerate(im):
        cumulative = np.max(im_series_ch,axis=0)
        if do_plot or do_plot:
            fig = plt.figure(figsize=(9, 9))
            plt.imshow(cumulative)
            plt.title(cn)
        if do_save:
            plt.savefig(os.path.join(destdir, filedesc + '%02d_max.png' % cn))
        if do_plot:
            plt.show()

    if as_colors is not None:
        cumulative = np.transpose(np.max(im[as_colors],axis=1),axes=[1,2,0])*16

        plt.imshow(cumulative)
        # plt.title(cn)
        # plt.savefig(os.path.join(destdir, filedesc + '_tifmax.png' % cn))
        print('cumulative:', cumulative.shape)
        plt.imshow(cumulative[:, :, 0])
        plt.show()
        plt.imshow(cumulative[:, :, 1])
        plt.show()
        plt.imshow(cumulative[:, :, 2])
        plt.show()
        imageio.imwrite(os.path.join(destdir, filedesc+ '_tifmax.png'),cumulative, format='PNG-FI')



# Max projection across all levels
def max_projection(globstring, do_plot=False, do_save=None, destdir=None, filedesc=None,as_colors=None):
    # filedir = 'HUVEC Tubulin AD 0.7 ZStack Decon cellSens.export/Intensity.png.export/'
    # destdir = 'HUVEC Tubulin AD 0.7 ZStack Decon cellSens.export/'
    # filedesc = 'HUVEC Tubulin AD 0.7 ZStack Decon cellSens'
    # globstring = os.path.join(filedir,'Intensity*.png')

    print('Options:  globstring, do_plot=False, do_save=None, destdir=None, filedesc=None,')
    filen = glob.glob(globstring)
    print(len(filen))

    cumulative = np.max(np.array([imageio.imread(_) for _ in filen]),axis=0)
    if as_colors is not None:
        cumulative = cumulative[:,:,as_colors]

    if do_plot:
        fig = plt.figure(figsize=(9, 9))
        plt.imshow(cumulative)
    if do_save:
        plt.imsave(os.path.join(destdir,filedesc+'_max.png'),cumulative)
        print('saved to ',os.path.join(destdir,filedesc+'_max.png'))
    if do_plot:
        plt.show()

class confocalObject(object):

    def __init__(self, quiet=False):

        self.data = {}
        self.quiet = quiet

    # filen to provide file list manually
    # globular to use glob.glob
    # ex_re are a list of regex used to extract metadata from the filenames (and the metadata target names)
    def load_images(self, target, filen=None, globular=None, index_re=r'(\d+)', third_index='auto'):

        if not self.quiet: print('cwd:', os.getcwd())

        if globular is not None:
            filen = glob.glob(globular)
            if len(filen)==0:
                print(glob.glob('*'))

        if not self.quiet:
            if not len(filen):
                print('No files found:', globular)
            print('Found %d files, ie:' % len(filen), filen[0])

        # produce indices, and
        filen, indices = self.sort_files(filen, index_re)

        if not self.quiet: print('Results: %d files w/ indices:' % len(filen), indices)

        if type(third_index) == int:
            self.data[target] = {index: imageio.imread(_)[:, :, third_index] for _, index in zip(filen, indices)}
        else:
            self.data[target] = {index: imageio.imread(_)[:, :] for _, index in zip(filen, indices)}

        # todo in this section:
        # load images in a way that allows us to associate them with an index. That's a good start

    # useful helper function for load_images
    def sort_files(self, filen, index_re):

        regex = re.compile(index_re)

        filen = sorted(filen, key=lambda x: int(regex.search(x.split('/')[-1]).group(1)))
        indices = map(lambda x: int(regex.search(x.split('/')[-1]).group(1)), filen)

        return filen, list(indices)

    # would be nice to use make_square, but it seems tricky
    def make_interp2D(self, target, result_tag=None, color=None, shrink=False, debug=False, make_square=True):

        if result_tag is None:
            result_tag = target + '_linear2D'

        def create_interp(data):
            x = np.arange(data.shape[0])
            y = np.arange(data.shape[1])

            xx, yy = np.meshgrid(x, y)  # not used?
            extent = (x[0], x[-1], y[0], y[-1])

            if shrink:
                foo = np.nonzero(data)
                foo2 = np.min(foo,axis=0)
                ea = np.array(
                    [np.min(np.nonzero(data)[0]), np.max(np.nonzero(data)[0]), np.min(np.nonzero(data)[1]), np.max(np.nonzero(data)[1])], dtype=int)

                x = x[ea[0]:ea[1]+1]
                y = y[ea[2]:ea[3]+1]

                new_extent = [x[0], x[-1], y[0], y[-1]]
                if debug: print('Shrinking extent from', extent,'to', new_extent)
                extent = new_extent
                data = data[ea[0]:ea[1]+1,ea[2]:ea[3]+1]



            return {
                'count': [len(x), len(y)],
                'extent': extent,
                'fxn': RegularGridInterpolator((x, y), data, bounds_error=False, fill_value=-1, method='linear')
            }

        if color is None:
            self.data[result_tag] = {idx: create_interp(self.data[target][idx]) for idx in self.data[target]}
        else:
            self.data[result_tag] = {idx: create_interp(self.data[target][idx][:, :, color]) for idx in
                                     self.data[target]}

    def make_interp3D(self, target, result_tag, color=None):

        if result_tag is None:
            result_tag = target + '_linear'

        def create_interp(data):
            # 3D interpolation must be regular grid. Ignore crazy Z extents
            indices = sorted(self.data[target].keys())
            if color is not None:
                data = np.array([self.data[target][_][:, :, color] for _ in indices]).transpose([1, 2, 0])
            else:
                data = np.array([self.data[target][_][:, :] for _ in indices]).transpose([1, 2, 0])

            x = np.arange(data.shape[0])
            y = np.arange(data.shape[1])
            z = np.linspace(np.min(indices), np.max(indices), len(indices))  # assumed!

            xx, yy = np.meshgrid(x, y)  # not used?
            extent = (x[0], x[-1], y[0], y[-1], z[0], z[-1])

            return {
                'count': [len(x), len(y), len(z)],
                'extent': extent,
                'fxn': RegularGridInterpolator((x, y, z), data, bounds_error=False, fill_value=0, method='linear')
            }

        if color is None:
            self.data[result_tag] = create_interp(self.data[target])
        else:
            self.data[result_tag] = create_interp(self.data[target])

    # Creates a cropped dataset in a new target name.
    def create_cropped(self, target, series=True, cropped_name=None, x0=None, window=None, debug=False, mask=None):

        def do_crop(item, window=None):
            if debug: print('item:', item)
            dims = len(item['extent']) // 2
            if debug: print('dims:', dims)

            if dims == 2:
                x = np.linspace(item['extent'][0], item['extent'][1], item['count'][0])
                y = np.linspace(item['extent'][2], item['extent'][3], item['count'][1])

                if window is None:
                    if mask is None:
                        if debug: print('Using extent as default window, because not provided')
                        window = item['extent']
                    else:
                        if debug: print('Using mask extent as default window')
                        window = mask['extent']
                x = x[(x >= window[0]) & (x <= window[1])]
                y = y[(y >= window[2]) & (y <= window[3])]

                xx, yy = np.expand_dims(x, 1), np.expand_dims(y,0)

                if mask is not None:
                    data = item['fxn']((xx, yy)) * mask['fxn']((xx, yy))
                else:
                    data = item['fxn']((xx, yy))

                if debug: print('New item has extent:', window)

                return {
                    'count': [len(x), len(y)],
                    'extent': window,
                    'fxn': RegularGridInterpolator((x, y), data, bounds_error=False, fill_value=0, method='linear')
                }

            elif dims == 3:
                x = np.linspace(item['extent'][0], item['extent'][1], item['count'][0])
                y = np.linspace(item['extent'][2], item['extent'][3], item['count'][1])
                z = np.linspace(item['extent'][4], item['extent'][5], item['count'][2])

                if window is None:
                    if mask is None:
                        if debug: print('Using extent as default window, because not provided')
                        window = item['extent']
                    else:
                        if debug: print('Using mask extent as default window')
                        window = [mask['extent'][0], mask['extent'][1], mask['extent'][2], mask['extent'][3], item['extent'][4], item['extent'][5], ]
                if len(window) < 6:
                    window = [window[0], window[1], window[2], window[3], item['extent'][4], item['extent'][5],]

                x = x[(x >= window[0]) & (x <= window[1])]
                y = y[(y >= window[2]) & (y <= window[3])]
                if len(window) > 4: z = z[(z >= window[4]) & (z <= window[5])]

                xx, yy, zz = np.reshape(x, [-1, 1, 1]), np.reshape(y, [1, -1, 1]), np.reshape(z, [1, 1, -1])

                if mask is not None:
                    data = item['fxn']((xx, yy, zz)) * mask['fxn']((xx, yy))
                else:
                    data = item['fxn']((xx, yy, zz))

                if debug:
                    print(xx.shape, yy.shape, zz.shape)
                    print(item['fxn']((x[0], y[0], z[0])))
                    print('New 3D item has extent:', window, 'counts:', [len(x), len(y), len(z)], )
                    print('z was:', z, item['count'][2])

                return {
                    'count': [len(x), len(y), len(z)],
                    'extent': window,
                    'fxn': RegularGridInterpolator((x, y, z), data, bounds_error=False, fill_value=0, method='linear')
                }

            else:
                raise Exception('wrong number of dims in cropped data', dims == 2, dims == 3, dims)

        data = self.data[target]
        # print('data keys:', data.keys())

        if series:
            if debug: print('series')
            self.data[cropped_name] = {idx: do_crop(data[idx], window=window) for idx in data}
        else:
            if debug: print('non-series')
            self.data[cropped_name] = do_crop(data, window=window)

    def render_image(self, target=None, data=None, do_plot=False, upscale=1, sliced=None, style='XY', series=False):

        if target is not None:
            data = self.data[target]

        if series: # 2D
            if sliced is None:
                raise Exception('Currently, must provide "sliced" to  render_image for 2D processing')
            else:
                data = data[sliced]
                if 'fxn' not in data:
                    raise Exception('Not sure what went wrong here, processing 2D data for render_image')

        if style == 'XY':
            ## XXXXXXX Not clear to me whey the other style plots shouldn't have a different expand_dims ordering too.
            px = np.expand_dims(np.linspace(data['extent'][0], data['extent'][1], data['count'][0] * upscale), 1)
            py = np.expand_dims(np.linspace(data['extent'][2], data['extent'][3], data['count'][1] * upscale), 0)
            rendered = data['fxn']((px, py)) if (sliced is None or series) else data['fxn']((px, py, sliced))
        if style == 'XZ':
            if series: raise Exception('cannot do XZ for series 2D data')
            px = np.expand_dims(np.linspace(data['extent'][0], data['extent'][1], data['count'][0] * upscale), 1)
            pz = np.expand_dims(np.linspace(data['extent'][4], data['extent'][5], data['count'][2] * upscale), 0)
            #             print('px', px, 'sliced at y:', sliced)
            rendered = data['fxn']((px, sliced, pz))
        if style == 'YZ':
            if series: raise Exception('cannot do YZ for series 2D data')
            py = np.expand_dims(np.linspace(data['extent'][2], data['extent'][3], data['count'][1] * upscale), 1)
            pz = np.expand_dims(np.linspace(data['extent'][4], data['extent'][5], data['count'][2] * upscale), 0)
            #             print('py', py, 'sliced at x:', sliced)
            rendered = data['fxn']((sliced, py, pz))

        if do_plot:
            plt.imshow(rendered.T, origin='lower', extent=data['extent'][0:4])

        return rendered

    # Currently not tested for 2D
    def get_weights(self, target):

        data = self.data[target]
        if 'fxn' in data:  # 3D

            levels = np.linspace(data['extent'][4], data['extent'][5], data['count'][2])
            weights = [np.sum(np.array(self.render_image(target=target, sliced=level)))
                       for level in levels]

            return levels, weights
        else:
            levels = sorted(data.keys())
            weights = [np.sum(np.array(self.render_image(data=data[level])))
                       for level in levels]

    # Works for 3D only at the moment
    def getCoM(self, target, debug=False):

        data = self.data[target]

        z, w = self.get_weights(target)
        if debug: print('Weights:', z, w)
        x0z = z[np.argmax(w)]
        if debug: print('x0z:', x0z)
        max_slice = self.render_image(target, sliced=x0z)
        cx, cy = center_of_mass(max_slice)
        if debug: print('but we need to adjust for extent')
        px = np.linspace(data['extent'][0], data['extent'][1], data['count'][0])
        py = np.linspace(data['extent'][2], data['extent'][3], data['count'][1])

        if debug: print(cx, cy)

        #     print()
        wxf = interp1d(np.arange(len(px)), px)
        wyf = interp1d(np.arange(len(py)), py)

        if debug: print(wxf(cx), wyf(cy))

        return (wxf(cx).item(), wyf(cy).item(), x0z)


    def find_envelope3D(self, target, result, n_angle=40, n_r=41, max_r=30, pct=0.95, force_int_z=True):

        x0 = self.getCoM(target)
        if force_int_z: x0 = [x0[0], x0[1], int(x0[2])]

        envelope = np.zeros((n_angle, n_angle, 3))

        rspace = np.linspace(0, max_r, n_r)
        phi_space = np.linspace(-np.pi / 2, np.pi / 2, envelope.shape[0])
        theta_space = np.linspace(0, 2 * np.pi, envelope.shape[1])

        def get_pct(x, pct=0.95):
            x_min = np.repeat(np.expand_dims(np.min(x, 0) * pct, 0), np.shape(x)[0], 0)
            return np.argmin(x_min - x < 0, 0)

        for i, phi in enumerate(phi_space):
            x0_repeats = np.array([[x0]]).repeat(len(theta_space), 1)
            x_nor = np.array([[[np.cos(th0) * np.cos(phi),
                                np.sin(th0) * np.cos(phi),
                                np.sin(phi)] for th0 in theta_space]])

            r_repeats = np.expand_dims(np.expand_dims(rspace, 1), 1)
            radial_xy = x0_repeats + x_nor * r_repeats

            indata = self.data[target]['fxn'](radial_xy)

            envelope[i] = np.array([radial_xy[arg, i, :]
                                    for arg, i in zip(get_pct(list(accumulate(-indata)), pct=pct),
                                                      range(len(np.argmin(list(accumulate(-indata)), 0))))])


        self.data[result] = envelope


    def find_radial_envelope(self, target, n_angle=40, n_r=101, max_r=60, sliced=None, pct=0.95):

        x0 = list(self.getCoM(target))
        if sliced is not None:
            x0[2] = sliced

        envelope = np.zeros((n_angle, n_angle, 3))

        rspace = np.linspace(0, max_r, n_r)
        theta_space = np.linspace(0, 2 * np.pi, envelope.shape[1])

        def get_pct(x, pct=pct):
            x_min = np.repeat(np.expand_dims(np.min(x, 0) * pct, 0), np.shape(x)[0], 0)
            return np.argmin(x_min - x < 0, 0)

        x0_repeats = np.array([[x0]]).repeat(len(theta_space), 1)
        x_nor = np.array([[[np.cos(th0),
                            np.sin(th0), 0] for th0 in theta_space]])

        r_repeats = np.expand_dims(np.expand_dims(rspace, 1),2)
        radial_xy = x0_repeats + x_nor * r_repeats

        indata = self.data[target]['fxn'](radial_xy)

        envelope = np.array([radial_xy[arg, i, :]
                                for arg, i in zip(get_pct(list(accumulate(-indata))),
                                                  range(len(np.argmin(list(accumulate(-indata)), 0))))])


        return envelope #

    def four_panel(self, target, do_plot=True, upscale=1, z=None, debug=False, slices=None, desc='', envelopes=[],
                   thickness=1, do_save=None, scatter_too=False, center_on=None, env_alpha=0.3, gaussian=None):

        wx, wy = self.get_weights(target)
        if z is not None:
            idx = int(np.where(wx == z)[0])
        else:
            idx = np.argmax(wy)
            z = int(wx[idx])

        extent = self.data[target]['extent']
        fig = plt.figure(figsize=(9, 9))
        ax1 = fig.add_subplot(221)

        # if slices is None: # deprecated?
        #     slice_xy, slice_yz, slice_xz = None, None, None

        # Easier definition if we just want to list target strings
        envelopes = [{'target':e,'color':'red'} if type(e)==str else e for e in envelopes]

        if slices is None:
            if center_on is not None:
                slice_yz, slice_xz, slice_xy = self.getCoM(center_on)
                if z is not None:
                    slice_xy = z
            else:
                print('slices is None')
                slice_yz = np.mean(extent[0:2]) # default slice: the midpoint of x
                slice_xz = np.mean(extent[2:4])
                slice_xy = z # 10 # FIXME
        else:
            slice_yz, slice_xz, slice_xy = slices[0], slices[1], z
            print('using given slices:', slice_yz, slice_xz, slice_xy)

        if debug: print('slices at: x=%0.1f, y=%0.1f, z=%0.1f' % (slice_yz, slice_xz, slice_xy))

        data_ax1 = self.render_image(target, sliced=slice_xy, upscale=upscale, do_plot=False)

        extent_xy = self.data[target]['extent'][0:4]
        if debug: print('showing extent:', extent)

        ############## XY

        to_show = data_ax1.T
        if gaussian is not None:
            to_show = gaussian_filter(to_show, gaussian[0])
        ax1.imshow(to_show, extent=extent_xy, origin='lower', aspect='auto')

        # Often the case that we want to restrict z to be integer [this mostly just shows up in title]
        apply_int_z = True
        mean_z = int(np.mean(extent[4:6])) if apply_int_z else np.mean(extent[4:6])

        # x0 = (np.mean(extent[0:2]), np.mean(extent[2:4]), mean_z)
        x0 = (slice_yz, slice_xz, slice_xy)
        if debug: print('extent:', extent, x0)
        ax1.axhline(y=x0[1], color='white', alpha=0.5, linestyle=':')
        ax1.axvline(x=x0[0], color='white', alpha=0.5, linestyle=':')

        ax1.set_ylabel('y axis (px)')
        # for xy plot, we can just use ph=0, the first element

        if len(envelopes) is not None:
            for nenv, envelope in enumerate(envelopes):

                env = self.data[envelope['target']]
                applies = abs(env[:, :, 2] - slice_xy) < thickness

                res = env[applies]
                if scatter_too:
                    ax1.scatter(res.T[0], res.T[1], color=envelope['color'], alpha=0.5)
                # print('res:', res)

                if not len(res):
                    print('Cannot do XY contour for env %d' % nenv)
                    continue
                hull = ConvexHull(res[:, [0,1]])
                for simplex in hull.simplices:
                    ax1.plot(res[simplex, 0], res[simplex, 1], color=envelope['color'], alpha=env_alpha)

        plt.title(desc + ' ' + str(x0))

        ################# XZ

        ax2 = fig.add_subplot(223, sharex=ax1)

        extent_xz = [extent[0], extent[1], extent[4], extent[5]]
        #             print('extent_xz:', extent_xz)

        data_ax2 = self.render_image(target, sliced=slice_xz, style='XZ', upscale=upscale, do_plot=False)

        to_show = data_ax2.T
        if gaussian is not None:
            to_show = gaussian_filter(to_show, gaussian[1])
        ax2.imshow(to_show, extent=extent_xz, origin='lower', aspect='auto')

        ax2.axhline(slice_xy, color='orange', alpha=0.5, linestyle=':')
        ax2.set_ylabel('z layer')
        ax2.set_xlabel('x axis (px)')

        if len(envelopes) is not None:
            for envelope in envelopes:
                env = self.data[envelope['target']]
                applies = abs(env[:, :, 1] - slice_xz) <= thickness
                res = env[applies]
                if scatter_too:
                    ax2.scatter(res.T[0], res.T[2], color=envelope['color'], alpha=0.5)

                if not len(res):
                    print('Cannot do XZ contour for env %d' % nenv)
                    continue

                hull = ConvexHull(res[:, [0,2]])
                for simplex in hull.simplices:
                    ax2.plot(res[simplex, 0], res[simplex, 2], color=envelope['color'], alpha=env_alpha)


        #################### YZ

        ax3 = fig.add_subplot(222, sharey=ax1)

        extent_yz = [extent[4], extent[5], extent[2], extent[3]]
        #             print('extent_yz:', extent_yz)


        data_ax3 = self.render_image(target, sliced=slice_yz, style='YZ', upscale=upscale, do_plot=False)

        to_show = data_ax3
        if gaussian is not None:
            to_show = gaussian_filter(to_show, gaussian[2])
        ax3.imshow(to_show, extent=extent_yz, origin='lower', aspect='auto')

        ax3.axvline(slice_xy, color='orange', alpha=0.5, linestyle=':')

        ax3.set_xlabel('z layer')

        if len(envelopes) is not None:
            for envelope in envelopes:
                env = self.data[envelope['target']]
                applies = abs(env[:, :, 0] - slice_yz) <= thickness
                res = env[applies]
                if scatter_too:
                    ax3.scatter(res.T[2], res.T[1], color=envelope['color'], alpha=0.5)

                if not len(res):
                    print('Cannot do YZ contour for env %d' % nenv)
                    continue

                hull = ConvexHull(res[:, [1,2]])
                if debug: print('shape:', res[:, [1,2]].shape, 'sliceyz:', slice_yz)
                for simplex in hull.simplices:
                    # print('simplex:')
                    ax3.plot(res[simplex, 2], res[simplex, 1], color=envelope['color'], alpha=env_alpha)


        ax4 = fig.add_subplot(224)

        wx, wy = self.get_weights(target)

        ax4.set_xlabel('Z Layer')
        ax4.scatter(z, wy[idx], marker='o', color='orange')
        ax4.axvline(z, color='orange', alpha=0.5, linestyle=':')
        ax4.set_yticklabels([])
        ax4.set_ylabel('image material (arb)')

        ax4.plot(wx, wy)

        if do_save is not None:
            # print('trying to save to:', do_save)
            fig.savefig(do_save,dpi=300)

def make_huvec26_obj(groupn=1):

    # print('Groupn is currently ignored')
    group_zstack_globular = "datasets/AFM-C HUVEC 20200626/Group %d.export/Intensity.png.export/Intensity_Z*_CH.png" % groupn
    group_nucleus_globular = "datasets/AFM-C HUVEC 20200626/Processing/objects/%d/IdentifyPrimaryObjects_*.png" % groupn
    group_cyto_globular = "datasets/AFM-C HUVEC 20200626/Processing/objects/%d/Cytoplasm_*.png" % groupn
    source_folder = 'datasets/AFM-C HUVEC 20200626/'

    cs1 = confocalObject(quiet=True)
    cs1.load_images(
        target='full_image',
        globular=group_zstack_globular,
        index_re=r'_Z(\d+)',
    )
    cs1.load_images(
        target='nucleus_mask',
        globular=group_nucleus_globular,
    )
    cs1.load_images(
        target='cyto_mask',
        globular=group_cyto_globular,
    )

    cs1.data['source_folder'] = source_folder
    cs1.make_interp2D('full_image', 'blue_linear2D', 2)  # nucleus
    cs1.make_interp2D('full_image', 'green_linear2D', 1)  # cyto
    cs1.make_interp2D('full_image', 'red_linear2D', 0)  # unused

    cs1.make_interp2D('nucleus_mask', 'nucleus_linear_mask2D', None, shrink=True)  # unused
    cs1.make_interp2D('cyto_mask', 'cyto_linear_mask2D', None, shrink=True)  # unused

    cs1.make_interp3D('full_image', 'cyto_linear3D', 1)  # nucleus 3D using color=2
    cs1.make_interp3D('full_image', 'nucleus_linear3D', 2)  # nucleus 3D using color=2
    cs1.make_interp3D('full_image', 'other_linear3D', 0)  # nucleus 3D using color=2

    return cs1


def make_default_obj(dataset_n=1, quiet=True):
    cs1 = confocalObject(quiet=quiet)

    if dataset_n == 1:
        # 'HUVEC Tubulin AD 0.7 ZStack Decon cellSens.export'
        globular_full = 'HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7 ZStack Decon cellSens.export/Intensity.png.export/Intensity*.png'
        globular_nucleus = 'HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7 ZStack Decon cellSens.export/objects/Nucleus*.png'
        globular_cyto = 'HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7 ZStack Decon cellSens.export/objects/Cytoplasm_*.png'
        source_folder = 'HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7 ZStack Decon cellSens.export/'

        nuc_color=2
        cyto_color=1
        other_color=0

    elif dataset_n ==2:
        # 'HUVEC Tubulin AD 0.7_decon_0002.png.frames' == red cyto + green nucleus
        globular_full = 'HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7_decon_0002.png.frames/HUVEC Tubulin AD 0.7_0002_Z*.png'
        globular_nucleus = 'HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7 ZStack Decon cellSens.export/objects/Nucleus*.png'
        globular_cyto = 'HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7 ZStack Decon cellSens.export/objects/Cytoplasm_*.png'
        source_folder = 'HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7_decon_0002.png.frames/'

        print('Full images n:', len(glob.glob(globular_full)))
        nuc_color=1
        cyto_color=0
        other_color=2

    elif dataset_n==3:
        # 'HUVEC Tubulin AD 0.7_decon_0002.png.frames' == red cyto + green nucleus
        globular_full = 'HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7_raw.export/Intensity.png.export/Intensity_Z*_CH.png'
        globular_nucleus = 'HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7 ZStack Decon cellSens.export/objects/Nucleus*.png'
        globular_cyto = 'HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7 ZStack Decon cellSens.export/objects/Cytoplasm_*.png'
        source_folder = 'HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7_raw.export/'

        print('Full images n:', len(glob.glob(globular_full)))
        nuc_color=2
        cyto_color=1
        other_color=0

    else:
        # 'HUVEC Tubulin AD 0.7_raw.export'
        raise Exception('Only 1 and 2. Other datasets not yet implemented')

    cs1.load_images(
        target='full_image',
        globular=globular_full,
        index_re=r'_Z(\d+)',
    )

    cs1.load_images(
        target='nucleus_mask',
        globular= globular_nucleus,
    )
    cs1.load_images(
        target='cyto_mask',
        globular= globular_cyto
    )

    cs1.data['source_folder'] = source_folder

    cs1.make_interp2D('full_image', 'blue_linear2D', 2)  # nucleus
    cs1.make_interp2D('full_image', 'green_linear2D', 1)  # cyto
    cs1.make_interp2D('full_image', 'red_linear2D', 0)  # unused

    cs1.make_interp2D('nucleus_mask', 'nucleus_linear_mask2D', None, shrink=True)  # unused
    cs1.make_interp2D('cyto_mask', 'cyto_linear_mask2D', None, shrink=True)  # unused

    cs1.make_interp3D('full_image', 'cyto_linear3D', cyto_color)  # nucleus 3D using color=2
    cs1.make_interp3D('full_image', 'nucleus_linear3D', nuc_color)  # nucleus 3D using color=2
    cs1.make_interp3D('full_image', 'other_linear3D', other_color)  # nucleus 3D using color=2

    ## Why don't these work for ds2?
    cs1.create_cropped('nucleus_linear3D', cropped_name='nucleus_cropped3D', series=False,
                             window=(360, 460, 350, 450, 1, 20), debug=True)
    cs1.create_cropped('cyto_linear3D', cropped_name='cyto_cropped3D', series=False,
                             window=(360, 460, 350, 450, 1, 20), debug=True)


    cs1.find_envelope3D('nucleus_cropped3D', 'nucleus_cropped_env')


    # Radial envelope;
    for cyto_id in sorted(cs1.data['cyto_linear_mask2D'].keys()):
        print('Cell:', cyto_id)
        cs1.create_cropped('cyto_linear3D', cropped_name='cyto_test_%02d' % cyto_id, debug=False, series=False,
                           mask=cs1.data['cyto_linear_mask2D'][cyto_id])
        cs1.create_cropped('nucleus_linear3D', cropped_name='nucleus_test_%02d' % cyto_id, debug=False, series=False,
                           mask=cs1.data['nucleus_linear_mask2D'][cyto_id])
        cs1.find_envelope3D('cyto_test_%02d' % cyto_id, 'cyto_cropped_env_%02d' % cyto_id, n_angle=320, max_r=200,
                            pct=0.95)
        cs1.find_envelope3D('nucleus_test_%02d' % cyto_id, 'nucleus_cropped_env_%02d' % cyto_id, n_angle=320, max_r=200,
                            pct=0.95)

        # cs1.four_panel('green_test_%02d' % cyto_id,
        #                envelopes=[{'target': 'green_cropped_env_%02d' % cyto_id, 'color': 'lightgreen'},
        #                           'blue_cropped_env_%02d' % cyto_id])
        # plt.show()
        break
    res = cs1.render_image('nucleus_test_01', sliced=5, do_plot=True, upscale=10)
    env01 = cs1.find_radial_envelope('nucleus_test_01')
    # print(cs1.data.keys())

    # This may just be the wrong idea here.
    # cs1.create_cropped('green_linear3D', cropped_name='green_test', debug=True, series=False, mask=cs1.data['cyto_linear_mask2D'][10])
    # cs1.four_panel('green_test')

    # print(cs1.data[''])

    # res = cs1.render_image('cyto_linear_mask2D', do_plot=True, sliced=20, series=True)

    if False:
        cs1 = cs1
        for cyto_id in sorted(cs1.data['cyto_linear_mask2D'].keys()):
            cs1.create_cropped('green_linear3D', cropped_name='green_test', debug=False, series=False,
                               mask=cs1.data['cyto_linear_mask2D'][cyto_id])
            cs1.create_cropped('blue_linear3D', cropped_name='blue_test', debug=False, series=False,
                               mask=cs1.data['nucleus_linear_mask2D'][cyto_id])
            cs1.find_envelope3D('green_test', 'green_cropped_env', n_angle=320, max_r=200, pct=0.95)
            cs1.find_envelope3D('blue_test', 'blue_cropped_env', n_angle=320, max_r=200, pct=0.95)

            cs1.four_panel('green_test',
                           envelopes=[{'target': 'green_cropped_env', 'color': 'lightgreen'}, 'blue_cropped_env'])
            # plt.show()

    return cs1

def make_good_stain_obj(quiet=False, make_max=False):

    cs1 = confocalObject(quiet=quiet)

    globular_full = 'datasets/HUVEC YAPr Centromere ZStack.export/Intensity.png.export/Intensity_Z*_CH.png'
    globular_nucleus = 'datasets/HUVEC YAPr Centromere ZStack.export/objects/Nucleus*.png'
    globular_cyto = 'datasets/HUVEC YAPr Centromere ZStack.export/objects/Cytoplasm_*.png'
    source_folder = 'datasets/HUVEC YAPr Centromere ZStack.export/'

    if make_max: max_projection(globular_full, do_plot=False, do_save=True, destdir=source_folder, filedesc='max_projection')



    print('Full images n:', len(glob.glob(globular_full)))
    nuc_color=2
    cyto_color=1
    other_color=0

    cs1.load_images(
        target='full_image',
        globular=globular_full,
        index_re=r'_Z(\d+)',
    )
    cs1.load_images(
        target='nucleus_mask',
        globular= globular_nucleus,
    )
    cs1.load_images(
        target='cyto_mask',
        globular= globular_cyto
    )

    cs1.data['source_folder'] = source_folder

    cs1.make_interp2D('full_image', 'blue_linear2D', 2)  # nucleus
    cs1.make_interp2D('full_image', 'green_linear2D', 1)  # cyto
    cs1.make_interp2D('full_image', 'red_linear2D', 0)  # unused

    cs1.make_interp2D('nucleus_mask', 'nucleus_linear_mask2D', None, shrink=True)  # unused
    cs1.make_interp2D('cyto_mask', 'cyto_linear_mask2D', None, shrink=True)  # unused

    cs1.make_interp3D('full_image', 'cyto_linear3D', cyto_color)  # nucleus 3D using color=2
    cs1.make_interp3D('full_image', 'nucleus_linear3D', nuc_color)  # nucleus 3D using color=2
    cs1.make_interp3D('full_image', 'other_linear3D', other_color)  # nucleus 3D using color=2


    ## Why don't these work for ds2?
    cs1.create_cropped('nucleus_linear3D', cropped_name='nucleus_cropped3D', series=False,
                             window=(360, 460, 350, 450, 1, 20), debug=True)
    cs1.create_cropped('cyto_linear3D', cropped_name='cyto_cropped3D', series=False,
                             window=(360, 460, 350, 450, 1, 20), debug=True)


    cs1.find_envelope3D('nucleus_cropped3D', 'nucleus_cropped_env')


    # Radial envelope;
    for cyto_id in sorted(cs1.data['cyto_linear_mask2D'].keys()):
        print('Cell:', cyto_id)
        cs1.create_cropped('cyto_linear3D', cropped_name='cyto_test_%02d' % cyto_id, debug=False, series=False,
                           mask=cs1.data['cyto_linear_mask2D'][cyto_id])
        cs1.create_cropped('nucleus_linear3D', cropped_name='nucleus_test_%02d' % cyto_id, debug=False, series=False,
                           mask=cs1.data['nucleus_linear_mask2D'][cyto_id])
        cs1.find_envelope3D('cyto_test_%02d' % cyto_id, 'cyto_cropped_env_%02d' % cyto_id, n_angle=320, max_r=200,
                            pct=0.95)
        cs1.find_envelope3D('nucleus_test_%02d' % cyto_id, 'nucleus_cropped_env_%02d' % cyto_id, n_angle=320, max_r=200,
                            pct=0.95)

        # cs1.four_panel('green_test_%02d' % cyto_id,
        #                envelopes=[{'target': 'green_cropped_env_%02d' % cyto_id, 'color': 'lightgreen'},
        #                           'blue_cropped_env_%02d' % cyto_id])
        # plt.show()
        break

    return cs1


# Loads a set of images using a default configuration.
def gen_load(o, use_masks=True):

    defaults = {
        'index_re': r'_Z(\d+)',
        'nuc_color': 2,
        'cyto_color': 1,
        'other_color': 0,
    }

    defaults.update(o)
    o = defaults

    cs1 = confocalObject(quiet=False)

    print('Full images n:', len(glob.glob(o['globular_full'])))

    cs1.load_images(
        target='full_image',
        globular=o['globular_full'],
        index_re=o['index_re']
    )

    if o['globular_nucleus'] is not None:
        cs1.load_images(
            target='nucleus_mask',
            globular=o['globular_nucleus'],
        )
    if o['globular_cyto'] is not None:
        cs1.load_images(
            target='cyto_mask',
            globular=o['globular_cyto'],
        )

    cs1.data['source_folder'] = o['source_folder']

    cs1.make_interp2D('full_image', 'blue_linear2D', 2)  # nucleus
    cs1.make_interp2D('full_image', 'green_linear2D', 1)  # cyto
    cs1.make_interp2D('full_image', 'red_linear2D', 0)  # unused

    if use_masks:
        cs1.make_interp2D('nucleus_mask', 'nucleus_linear_mask2D', None, shrink=True)  # unused
        cs1.make_interp2D('cyto_mask', 'cyto_linear_mask2D', None, shrink=True)  # unused

    cs1.make_interp3D('full_image', 'cyto_linear3D', o['cyto_color'])  # nucleus 3D using color=2
    cs1.make_interp3D('full_image', 'nucleus_linear3D', o['nuc_color'])  # nucleus 3D using color=2
    cs1.make_interp3D('full_image', 'other_linear3D', o['other_color'])  # nucleus 3D using color=2

    cs1.data['gen_load_options'] = o

    return cs1


def export_annotated_SR1():


    annotation_folder = 'datasets/SR1/Export - Original Channel tiff Stack/annotation'

    tags = ['nucleus_linear3D', 'cyto_linear3D', 'other_linear3D']

    envelope_nucleus = contour.convert_annotated(annotation_folder, 'nucleus_linear3D')
    envelope_cyto = contour.convert_annotated(annotation_folder, 'cyto_linear3D')
    envelope_other = contour.convert_annotated(annotation_folder, 'other_linear3D')

    # print(res)

    envelope_nucleus, new_center = heal_contour(envelope_nucleus, do_connect=True, doPlot=False)
    envelope_cyto, _ = heal_contour(envelope_cyto, move_to_mean=new_center, do_connect=True, doPlot=False)
    envelop_other, _ = heal_contour(envelope_other, move_to_mean=new_center, do_connect=True, doPlot=False)

    export_folder = os.path.join(annotation_folder, 'export')
    os.makedirs(export_folder,exist_ok=True)

    scaling = [0.03237, 0.03237, 0.1]
    cyto_meta = export_contours(envelope_cyto, ncvar='NCY', filen='C_Cyto_Z', scaling=scaling,
                                         export_dir=export_folder, color_range='green',
                                         move_to_mean=None)
    nuc_meta = export_contours(envelope_nucleus, ncvar='NC', filen='C_Nuc_Z', scaling=scaling,
                                        export_dir=export_folder, color_range='red',
                                        move_to_mean=cyto_meta['move_to_mean'])

    meta_fname = os.path.join(export_folder, 'file.txt')

    export_contours_meta('SR1_test', meta_fname, cyto_meta, nuc_meta)


# Exports images for use with labelme
def export_set_for_annotation(cs1, tag=None, use_log=False, do_show=False, do_parallel=False, vmax=200):
    if tag is None:
        print('Need tag: cyto_linear3D, nucleus_linear3D, other_linear3D')

    print('source:', cs1.data['source_folder'])

    annotation_folder = os.path.join(cs1.data['source_folder'], 'annotation')
    os.makedirs(annotation_folder, exist_ok=True)

    ranged = np.array(cs1.data[tag]['extent'])
    print(cs1.data[tag]['extent'])
    print(ranged[2], ranged[5])
    if do_parallel:
        param_list = [(cs1.render_image(tag, sliced=int(f), series=False, do_plot=False),
                       tag, f, use_log, annotation_folder, do_show, ranged[5]-1) for f in tqdm(range(int(ranged[2]), int(ranged[5])))]
        print('Starting parallel save:', range(int(ranged[2]), int(ranged[5])))
        with mp.Pool() as p:
            res = p.map(export_single,param_list)

    else:
        for f in range(int(ranged[2]), int(ranged[5])):
            sliced = cs1.render_image(tag, sliced=int(f), series=False, do_plot=False)



            if use_log:
                plt.imshow(np.log(sliced + 1))
            else:
                plt.imshow(sliced)

                if vmax is not None: plt.set_clim(0,vmax)
            plt.title(tag + ' ' + str(f))

            suffix = '_%03d.png' if ranged[5]-1>99 else '_%02d.png'

            figpath = os.path.join(annotation_folder, tag + suffix % f)
            print('figpath:', figpath)
            plt.savefig(figpath, bbox_inches='tight', dpi=300)
            if do_show: plt.show()
            plt.clf()



def export_single(params):
    sliced, tag, f, use_log, annotation_folder, do_show, max_f = params

    if use_log:
        plt.imshow(np.log(sliced + 1))
    else:
        plt.imshow(sliced)
    plt.title(tag + ' ' + str(f))
    suffix = '_%02d.png' if max_f < 99 else '_%03d.png'
    figpath = os.path.join(annotation_folder, tag + suffix % f)
    print('figpath:', figpath)
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    if do_show: plt.show()
    plt.clf()


def export_1(path_tif, path_export, do_split=True):

    base_dir = os.path.dirname(path_tif)
    target_dir = path_export
    source_folder = os.path.dirname(path_tif)

    head, tail = os.path.split(path_tif)

    # export_path = os.path.join(target_dir,'exported')
    path_exported = target_dir # os.path.join(target_dir,'exported')

    if do_split: export_sr_ome(path_exported, source_folder, tail)

    globular_full = os.path.join(path_export,'Intensity*.png')  # 'AFM Test 100x/AFM Oversampling - Deconvolution/Intensity.png.export/*.png'
    print('checking for ', globular_full)
    print(len(glob.glob(globular_full)))
    # os.makedirs(os.path.join(destdir, 'objects'), exist_ok=True)
    if True:
        do_plot = False
        max_projection(globular_full, do_plot=do_plot, do_save=True, destdir=target_dir,
                                filedesc='projection', as_colors=[0, 1, 2])
        if do_plot: plt.show()


def export_2(path_tif, path_export, use_masks=False):
    example_z = 4

    base_folder, tail = os.path.split(path_tif)
    path_exported = path_export # os.path.join(path_export,'exported')

    objects_folder_base = base_folder  # 'dataset/20200724 AFM Test 3/HUVEC Cyto(g) Nucleus(b).export'
    #     objects_folder_base = 'datasets/HUVEC 3D Reconstruction Files/HUVEC Tubulin AD 0.7 ZStack Decon cellSens.export/'

    globular_full = os.path.join(path_exported,
                                 'Intensity_Z*.png')  # 'AFM Test 100x/AFM Oversampling - Deconvolution/Intensity.png.export/*.png'
    print('Raw png count:', globular_full, len(glob.glob(globular_full)))
    # destdir = base_folder
    # desc = 'sr3B'
    # 'index_re': r'_Z(\d+)',
    o = dict(
        globular_full=globular_full,
        # reuse segmentation
        globular_nucleus=None, #os.path.join(objects_folder_base, 'objects/Nucleus*.png'),
        globular_cyto=None, #os.path.join(objects_folder_base, 'objects/Cytoplasm*.png'),
        source_folder=path_exported,  # diagnostics folder will be created here

        nuc_color=0,  # 2 0 2
        cyto_color=1,
        other_color=2,
    )

    cs1 = gen_load(o, use_masks=use_masks)
    print(cs1.data.keys())


    return cs1

def export_3(cs1, export_path):

    example_z = 4
    if False:
        res = cs1.render_image('cyto_linear3D', sliced=example_z, series=False, do_plot=False)
        plt.imshow(gaussian_filter(res, 0))
        plt.show()

    if False:
        ww = np.array(cs1.get_weights('cyto_linear3D'))
        plt.plot(ww[0], ww[1])
        plt.show()

    if True:
        cs1.create_cropped('cyto_linear3D', cropped_name='cyto_test', debug=False, series=False)
        print('cropped cyto')
        cs1.create_cropped('nucleus_linear3D', cropped_name='nucleus_test', debug=False, series=False)
        print('cropped nuc')
        cs1.create_cropped('other_linear3D', cropped_name='other_test', debug=False, series=False)
        print('cropped  other')
        print('finished cropping')

        if True:
            cs1.four_panel('cyto_test',desc='AFM100-N1', center_on='nucleus_test', z=example_z,upscale=1,
                           do_save=os.path.join(export_path, 'four_fig.png')) # upscale=1.0, z=13,
            print('finished four_panel')
            plt.show()
        #     cs1.four_panel('cyto_test',desc='AFM100-N1')

    if True:
        export_set_for_annotation(cs1, 'cyto_linear3D', use_log=False, do_show=False)
        export_set_for_annotation(cs1, 'other_linear3D', use_log=False, do_show=False)
        export_set_for_annotation(cs1, 'nucleus_linear3D', use_log=False, do_show=False)


def loading_SR_HUVEC1():
    base_dir = 'datasets/SR_HUVEC1/'
    specific_ds = 'HUVEC Cell 1/'
    target_dir = os.path.join(base_dir, specific_ds, 'Intensity.png.export/')
    source_folder = os.path.join(base_dir, specific_ds)

    #     print('isfile', os.path.isfile(os.path.join(source_folder,'HUVEC Cell 1.ome.tif')))
    if True:
        export_sr_ome(target_dir, source_folder, 'HUVEC Cell 1.ome.tif', n_channels=4, keep3=[0,2,3]) # 0, 1, 3

    globular_full = os.path.join(base_dir, specific_ds,
                                 'Intensity.png.export/Intensity*.png')  # 'AFM Test 100x/AFM Oversampling - Deconvolution/Intensity.png.export/*.png'
    print(len(glob.glob(globular_full)))
    destdir = os.path.join(base_dir, specific_ds)
    os.makedirs(os.path.join(destdir, 'objects'), exist_ok=True)
    desc = 'srh1_1'
    if True:
        max_projection(globular_full, do_plot=True, do_save=True, destdir=destdir,
                                filedesc=desc, as_colors=[0, 1, 2])
    plt.show()

def build_export_contour():

    # specify contour folder
    # Load contours
    # Process contours
    # Export files
    # Export validation images


    pass

if __name__ == '__main__':

    if False:
        parser = argparse.ArgumentParser(description='Export png from ome tif')
        parser.add_argument('--source_tif', required=True, help='path to tif file')
        parser.add_argument('--out_dir', default='exported', help='path to export')

        args = vars(parser.parse_args())
        print(args)

        export_1(args['source_tif'], args['out_dir'], do_split=True)
        cs1 = export_2(args['source_tif'], args['out_dir'])
        export_3(cs1, args['out_dir'])
        # --source_tif="datasets/SR3/HUVEC Geometry 3_Reconstructed_crop/HUVEC Geometry 3_Reconstructed_crop_HUVEC Geometry 3_Reconstructed_crop.ome.tif"

    if False:
        loading_SR_HUVEC1()

    pass
