import glob
from tqdm import tqdm
import json
import numpy as np
import itertools
from scipy.interpolate import interp1d
import os
import cv2
import matplotlib.pyplot as plt

def convert_annotated(target_dir,tag):
    graded = {}
    fnames = list(sorted(glob.glob(os.path.join(target_dir, tag + '*.json'))))

    for fname in tqdm(fnames):
        data = json.load(open(fname))
        #         print(len(fnames), fnames[0] if len(fnames) else 'None')
        level = int(fname.split('_')[-1].split('.json')[0])
        # print('tag: {} / '.format(tag) + 'file:', level, len(data['shapes'][0]['points']))

        graded[level] = np.array(data['shapes'][0]['points']).T

    return graded


def reparam_contour3(c, guide_contour=None, npts=21, debug=False, doPlot=False, padding=0):
    for level in sorted(c.keys()):

        if guide_contour is not None and level in guide_contour:
            guide_level = guide_contour[level]
        else:
            #             print('No guide')
            guide_level = None

        #         print('c[level]', c[level])
        if len(c[level]):
            edited_level = edit_level(c[level], guide_level, npts, debug=debug, leveln=level, doPlot=doPlot,
                                      padding=padding)
            c[level] = np.array(edited_level)

    return c

# polar version of reparam_contour3. Requires recentering.
def reparam_contour4(c, guide_contour=None, npts=21, debug=False, doPlot=False, padding=0, recentering=None):

    for level in sorted(c.keys()):

        if guide_contour is not None and level in guide_contour:
            guide_level = guide_contour[level]
        else:
            #             print('No guide')
            guide_level = None

        #         print('c[level]', c[level])
        if len(c[level]):
            # print('edit_level_polar invocation')
            edited_level = edit_level_polar(c[level], guide_level, npts, debug=debug, leveln=level, doPlot=doPlot,
                                      padding=padding, recentering=recentering)
            c[level] = np.array(edited_level)

    return c



def edit_level(c, guide_c, npts=21, debug=False, leveln=-1, doPlot=False, padding=0):

    # Walks along two contours, interpolating more points
    def make_more_pts_dist(contour, cX, cY, heal=False):

        # Get points from this level
        lx = contour[0]
        ly = contour[1]

        if heal:
            #             print('healing!', lx)
            lx = np.hstack([lx, [lx[0], ]])
            ly = np.hstack([ly, [ly[0], ]])

            # Accumulate distance along the contour described, so that we can interpolate along it
        dist = np.array(
            [0, ] + [np.sqrt((i1 - i0) ** 2 + (j1 - j0) ** 2) for i0, i1, j0, j1 in zip(lx[1:], lx, ly[1:], ly)])
        sumdist = np.array(list(itertools.accumulate(dist)))

        # Interpolation of x, y after distance sumdist along the curve
        fx = interp1d(sumdist, lx)
        fy = interp1d(sumdist, ly)

        # Interpolate 101 pieces
        gd = np.linspace(0, sumdist[-1], npts)
        gx_, gy_ = fx(gd), fy(gd)

        return gx_, gy_

    #


    #     print('Editing level:', leveln, guide_c is not None)

    #     print(c)
    M = cv2.moments(c.T.astype(np.int32))
    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    # Fill in points and generate radius/theta interpolation
    gx, gy = make_more_pts_dist(c, cX=cX, cY=cY, heal=True)

    #     plt.show()

    r, t, ftr1 = get_radius_theta(gx, gy, cX=cX, cY=cY)
    gx, gy = cX + r * np.cos(t), cY + r * np.sin(t)

    if guide_c is not None:

        # flesh out guide contour
        Mg = cv2.moments(guide_c.T.astype(np.int32))
        cXg, cYg = int(Mg["m10"] / Mg["m00"]), int(Mg["m01"] / Mg["m00"])
        ggx, ggy = make_more_pts_dist(guide_c, cX=cXg, cY=cYg, heal=True)
        #         plt.plot(guide_c[0], guide_c[1])
        rg, tg, ftr1g = get_radius_theta(ggx, ggy, cX=cXg, cY=cYg)

        if True:
            # Replace guide contour with theta points
            t0g = np.linspace(-np.pi, np.pi - np.pi / npts, npts)
            r0g = ftr1g(t0g)
            #         gx, gy = cXg+r*np.cos(t), cYg + r*np.sin(t)
            ggx, ggy = cXg + r0g * np.cos(t0g), cYg + r0g * np.sin(t0g)

        if doPlot:
            plt.scatter(ggx, ggy)
            plt.plot(ggx, ggy)

        gx, gy = make_more_pts_dist(c, cX=cXg, cY=cYg, heal=True)
        r, t, ftr1 = get_radius_theta(gx, gy, cX=cXg, cY=cYg)

        if True:
            # Replace contour with theta points
            t0 = np.linspace(-np.pi, np.pi - np.pi / npts, npts)
            r0 = ftr1(t0)
            #         gx, gy = cXg+r*np.cos(t), cYg + r*np.sin(t)
            gx, gy = cXg + r0 * np.cos(t0), cYg + r0 * np.sin(t0)

            # expand contour
            #         print(r0-r0g)

            if padding is not None:
                diff = np.array(r0 - r0g)
                r0 = np.array(r0)
                #             print('edit:', -diff*(diff<0))
                #             padding = 10
                r0 += -(diff - padding) * (diff < padding)
                gx, gy = cXg + r0 * np.cos(t0), cYg + r0 * np.sin(t0)
    #         print(t - tg)

    # remake real contour with guide's center

    if doPlot:
        plt.plot(gx, gy)
        plt.scatter(gx, gy)
        plt.show()

    return [gx, gy]

# polar version of edit_level_polar
def edit_level_polar(c, guide_c, npts=21, debug=False, leveln=-1, doPlot=False, padding=0, recentering=None):

    #     print(c)
    if recentering is None:
        M = cv2.moments(c.T.astype(np.int32))
        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        # print('Had to find recentering:', (cX, cY))
    else:
        # print('Used provided recenter:', recentering)
        cX, cY = recentering

    # Fill in points and generate radius/theta interpolation
    gx, gy = make_more_pts_dist_edited(c, heal=True)

    #     plt.show()

    r, t, ftr1 = get_radius_theta(gx, gy, cX=cX, cY=cY)
    t0 = np.linspace(-np.pi, np.pi - np.pi / npts, npts)
    r0g = ftr1(t0)

    gx, gy = cX + r0g * np.cos(t0), cY + r0g * np.sin(t0)
    # gx, gy = cX + r * np.cos(t), cY + r * np.sin(t)

    if guide_c is not None:

        # flesh out guide contour
        # Mg = cv2.moments(guide_c.T.astype(np.int32))
        # cXg, cYg = int(Mg["m10"] / Mg["m00"]), int(Mg["m01"] / Mg["m00"])
        cXg, cYg = cX, cY

        ggx, ggy = make_more_pts_dist_edited(guide_c, heal=True)
        #         plt.plot(guide_c[0], guide_c[1])
        rg, tg, ftr1g = get_radius_theta(ggx, ggy, cX=cXg, cY=cYg)

        if True:
            # Replace guide contour with theta points
            t0g = np.linspace(-np.pi, np.pi - np.pi / npts, npts)
            r0g = ftr1g(t0g)
            #         gx, gy = cXg+r*np.cos(t), cYg + r*np.sin(t)
            ggx, ggy = cXg + r0g * np.cos(t0g), cYg + r0g * np.sin(t0g)

        if doPlot:
            plt.scatter(ggx, ggy)
            plt.plot(ggx, ggy)
            plt.scatter(cXg, cYg,marker='8', color='black')
            plt.title('Polar')

        gx, gy = make_more_pts_dist_edited(c, heal=True)
        r, t, ftr1 = get_radius_theta(gx, gy, cX=cXg, cY=cYg)

        if True:
            # Replace contour with theta points
            t0 = np.linspace(-np.pi, np.pi - np.pi / npts, npts)
            r0 = ftr1(t0)
            #         gx, gy = cXg+r*np.cos(t), cYg + r*np.sin(t)
            gx, gy = cXg + r0 * np.cos(t0), cYg + r0 * np.sin(t0)

            # expand contour
            #         print(r0-r0g)

            if padding is not None:
                diff = np.array(r0 - r0g)
                r0 = np.array(r0)
                #             print('edit:', -diff*(diff<0))
                #             padding = 10
                r0 += -(diff - padding) * (diff < padding)
                gx, gy = cXg + r0 * np.cos(t0), cYg + r0 * np.sin(t0)
    #         print(t - tg)

    # remake real contour with guide's center

    if doPlot:
        plt.plot(gx, gy)
        plt.scatter(gx, gy)
        plt.title('Polar 2')
        plt.scatter(cX, cY, marker='8', color='black')
        plt.show()

    return [gx, gy]



def tip_to_realspace(tip, CoM, debug=False):

    res = {}
    x, y = tip

    res['orig'] = (x,y)

    # mirror the axes:
    # x = 2048 - x # x is not mirrored.
    y = 2048 - y

    res['mirrored'] = (x,y)

    if debug: print('Mirrored:', x, y)

    x *= np.mean([910.8118132964248, 910.8118132964248, 908.1081081081081]) / 2048.0
    y *= np.mean([910.8118132964248, 910.8118132964248, 908.1081081081081]) / 2048.0

    # add the annotation offset
    # x += (160.39189189 + 169.17567568) / 2.0
    x += (163.72972973 + 165.08108108) / 2.0
    # y += (92.63513514 + 95.67567568) / 2.0
    y += (90.94594595 + 90.94594595) / 2.0

    res['annotated'] = (x,y)

    if debug: print('On annotation graph:', x, y)

    # move to CoM
    x -= CoM[0]
    y -= CoM[1]
    if debug: print('After CoM shift:', x, y)

    res['shifted'] = (x,y)

    # scale from annotation to real micron space
    annotation_scaling = np.mean([910.8118132964248, 910.8118132964248, 908.1081081081081]) / 2048.0
    scaling_factor = 0.03237 / annotation_scaling

    x_real = x * scaling_factor
    y_real = y * scaling_factor

    # real space (for ansys) and contour space (for plotting)
    res['final'] = (x_real, y_real)
    return res

def get_radius_theta(x, y, cX, cY):
    dX = x - cX
    dY = y - cY

    radius = np.sqrt(dX ** 2 + dY ** 2)
    theta = np.arctan2(dY, dX)
    #         print('theta:', theta)

    #         ftx_ = interp1d(theta, dX)
    #         fty_ = interp1d(theta, dY)

    st, sr = zip(*sorted(zip(theta, radius)))

    #         print('theta interp:', st, sr)

    ftr = interp1d(st, sr, bounds_error=False, fill_value=(sr[0] + sr[-1]) / 2)

    #         print('funciont theta:', st)

    return radius, theta, ftr

def get_scaling_factor(fname_scaling):
    # scaling factor
    scaling_json = json.load(fname_scaling)
    scaling_points = np.array(scaling_json['shapes'][0]['points'])
    s0 = scaling_points[0]
    ds = [np.sqrt((nex[0] - pre[0]) ** 2 + (nex[1] - pre[1]) ** 2) for nex, pre in
          zip(scaling_points[1:], scaling_points[0:-1])]
    print(scaling_points)
    # ds = [nex for nex,pre in zip(scaling_points[1:],  scaling_points[0:-1])]
    print(ds, np.mean(ds) / 2048.0)
    scaling_factor_dots_per_px = np.mean(ds) / 2048.0
    sf = scaling_factor_dots_per_px

    print('sf', sf)

    return sf

def transform_probe(p, moved=None, move_to_mean=None, debug=True, mirror_x=False, mirror_y=False):

    print('debug:', debug)
    probe_pred = np.array(p)
    if mirror_x:
        probe_pred[0] = ((2048 - probe_pred[0]) * ((1074 - 165) / 2048)) + 165
    else:
        probe_pred[0] = ((2048 * 0 + probe_pred[0]) * ((1074 - 165) / 2048)) + 165

    if mirror_y:
        probe_pred[1] = ((2048 - probe_pred[1]) * ((1000 - 90) / 2048)) + 90
    else:
        probe_pred[1] = ((2048 * 0 + probe_pred[1]) * ((1000 - 90) / 2048)) + 90

    if moved is not None:
        probe_pred = probe_pred - moved
        if debug: print('subtracting moved of', moved,'to', probe_pred )

    if move_to_mean is not None:
        probe_pred[0:2] -= move_to_mean[0:2]
        if debug: print('subtracting move_to_mean of', move_to_mean[0:2],'to', probe_pred )


    return probe_pred

def export_probes(probes, export_dir, moved=None, move_to_mean=None, scaling=1):

    with open(os.path.join(export_dir, 'probe_cyto.txt'),'w') as fout:

        p = probes[0]
        if moved is not None:
            p -= moved

        if move_to_mean is not None:
            p -= move_to_mean[0:2]

        p *= scaling
        fout.write('\n'.join(['%0.3f' % p[0], '%0.3f' % p[1]]))

    with open(os.path.join(export_dir, 'probe_nuclear.txt'),'w') as fout:

        p = probes[1]
        if moved is not None:
            p -= moved
            print('Moved was', moved,'to', p)

        if move_to_mean is not None:
            p -= move_to_mean[0:2]
            print('Move_to_mean was', move_to_mean,'to', p)

        p *= scaling
        fout.write('\n'.join(['%0.3f' % p[0], '%0.3f' % p[1]]))

def load_exported(fname):

    with open(fname,'r') as fin:
        res = fin.readlines()

    # print(res)
    res = np.array([_.strip().split(',')[2:4] for _ in res if len(_) and _[0]=='k'], dtype=np.float)

    return res

def load_exported_ext(fname):

    with open(fname,'r') as fin:
        res = fin.readlines()

    # print(res)
    res = np.array([_.strip().split(',')[2:] for _ in res if len(_) and _[0]=='k'], dtype=np.float)

    return {
        'rows':res,
        'other': None,
        }

# lightly modified to run, from contour.py, edit_level fxn
def make_more_pts_dist_edited(contour, heal=False, npts=101):

    # Get points from this level
    lx = contour[0]
    ly = contour[1]

    if heal:
        #             print('healing!', lx)
        lx = np.hstack([lx, [lx[0], ]])
        ly = np.hstack([ly, [ly[0], ]])

        # Accumulate distance along the contour described, so that we can interpolate along it
    dist = np.array(
        [0, ] + [np.sqrt((i1 - i0) ** 2 + (j1 - j0) ** 2) for i0, i1, j0, j1 in zip(lx[1:], lx, ly[1:], ly)])
    sumdist = np.array(list(itertools.accumulate(dist)))

    # Interpolation of x, y after distance sumdist along the curve
    fx = interp1d(sumdist, lx)
    fy = interp1d(sumdist, ly)

    # Interpolate 101 pieces
    gd = np.linspace(0, sumdist[-1], npts)
    gx_, gy_ = fx(gd), fy(gd)

    return gx_, gy_

def smoothed(r, window=3, use_log=False):
    if use_log: r = np.log(r)
    res = np.convolve(r, np.ones(window), 'same') / np.convolve(r * 0 + 1, np.ones(window), 'same')
    if use_log: res = np.exp(res)

    return res


# Interpolate and enforce some monotonicity in the values of along a polar ray
def make10_nondec(r, levels, n=10, levels_predicted=None, do_smooth=True, nondec=True, minmax='min',
                  trim=True, guide=None):
    if levels_predicted is None:
        levels_predicted = [round(_,2) for _ in np.linspace(levels[0], levels[-1], 10)]

    if do_smooth: r = smoothed(r, window=3)  # apply moving average

    r_predicted = np.interp(levels_predicted, levels, r, left=None, right=None)

    if nondec:
        r_predicted = np.minimum.accumulate(r_predicted) if minmax == 'min' else np.maximum.accumulate(r_predicted)

    return levels_predicted, r_predicted

# r1 is cyto, r2 is nucleus (guide)
def bulge(r1, z1, r2, z2, scaling, bb=.020, debug=False):
    if scaling is None:
        # print('Warning: Using predefined scaling...',  (0.07285744510047523, 0.07285744510047523, 0.1))
        scaling = (0.07285744510047523, 0.07285744510047523, 0.1)

    newz, newr = [], []

    for z, r in zip(z1, r1):  # for each outer point, displaced by rp

        ginterpdr = np.interp([z], z2, r2)[0]

        rp = max(r, ginterpdr + int(bb/scaling[0]))

        newz.append(z)
        newr.append(rp)

    newzi = np.linspace(newz[0], newz[-1], 10)
    newzi = [round(_,2) for _ in newzi]

    newri = np.interp(newzi, newz, newr)
    return newzi, newri
    # return newz, newr


def pplinecuts(nuc_padded, cyto_padded, recentering, nangles=20, nlevs=10, bb=10, doPlot=False, savePlot=None, scaling=None):
    new_nuc = []
    new_cyto = []
    # Conver plotting to side-by-side comparison
    for polar_anglen in range(nangles):  # shouldn't hardcode number

        levs = sorted(nuc_padded.keys())
        desired_levels = np.linspace(levs[0], levs[-1], nlevs)  # shouldn't hardcode number
        desired_levels = [round(_, 2) for _ in desired_levels]

        x = np.array([nuc_padded[lev][0][polar_anglen] - recentering[0] for lev in levs])
        y = np.array([nuc_padded[lev][1][polar_anglen] - recentering[1] for lev in levs])
        #     plt.plot(x,y)

        levsp, rpp = make10_nondec(np.sqrt(x ** 2 + y ** 2), levs, levels_predicted=desired_levels, do_smooth=True,
                                   nondec=False)
        levspg, rppg = levsp, rpp  # save as guide

        new_nuc.append([levsp, rpp])

        if doPlot or savePlot:
            fig, axs = plt.subplots(1, 2)

            ax0, ax1 = axs

            ax0.scatter(rpp, levsp, color='blue')
            ax0.plot(np.sqrt(x ** 2 + y ** 2), levs, color='blue', linestyle=':')
            ax0.set_title('Angle #%d' % polar_anglen)
            ax0.set_ylabel('Z (px)')
            #

            ax1.scatter(rpp, levsp, color='blue')
            ax1.plot(np.sqrt(x ** 2 + y ** 2), levs, color='blue', linestyle=':')
            ax1.set_title('Angle #%d' % polar_anglen)
            ax1.set_ylabel('Z (px)')
        #     plt.show()

        levs = sorted(cyto_padded.keys())
        desired_levels = np.linspace(levs[0], levs[-1], nlevs)
        desired_levels = [round(_, 2) for _ in desired_levels]

        x = np.array([cyto_padded[lev][0][polar_anglen] - recentering[0] for lev in levs])
        y = np.array([cyto_padded[lev][1][polar_anglen] - recentering[1] for lev in levs])
        #     plt.plot(x,y)

        levsp, rpp = make10_nondec(np.sqrt(x ** 2 + y ** 2), levs, levels_predicted=desired_levels, do_smooth=True)
        #     levspg, rppg = levsp, rpp # save as guide

        if doPlot or savePlot:
            ax0.scatter(rpp, levsp, color='red')
            ax0.plot(np.sqrt(x ** 2 + y ** 2), levs, color='red', linestyle=':')

        levspb, rppb = bulge(rpp, levsp, rppg, levspg, scaling=scaling, bb=bb)

        new_cyto.append([levspb, rppb])

        if doPlot or savePlot:
            #         plt.scatter(rpp,levsp, color='red')

            #         plt.plot(np.sqrt(x**2 + y**2),levs, color='red',linestyle=':')
            ax1.plot(rppb, levspb, color='red')  # ,linestyle=':')

            #     plt.xscale('log')

            ax1.set_xlabel('r (px)')
            ax1.set_ylabel('z (slice)')

        if savePlot:
            fig.savefig(os.path.join(savePlot,'ang_%d' % polar_anglen))
        if doPlot:
            fig.show()

        if savePlot or doPlot:
            plt.close(fig)


    return np.array(new_nuc).transpose(1, 2, 0), np.array(new_cyto).transpose(1, 2, 0)

# returns objects to xy space dicts
def re_xy(obj, npts=11):
    #     print(obj.shape, )

    th = np.linspace(-np.pi, np.pi - np.pi / npts, npts)

    obj_xy = {
        obj[0, ii, 0]: np.array([
            obj[1, ii, :] * np.cos(th),
            obj[1, ii, :] * np.sin(th)])
        for ii in range(obj.shape[1])
    }  # cXg + r0g * np.cos(t0g), cYg + r0g * np.sin(t0g)

    return obj_xy
