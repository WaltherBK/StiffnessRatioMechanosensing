# make_axial_features.py

import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from pyefd import elliptic_fourier_descriptors
import pyefd
import cv2 as cv
from scipy.spatial import ConvexHull, convex_hull_plot_2d

BASE_DIR = 'export_contours/'
OUTPUT_DIR = '/Users/asears/work/confocal/axial_features/'

with open(os.path.join(OUTPUT_DIR,f'axial_summary.txt'),'w') as fout:
    fout.write('# mean of the (solidity, convexity, circularity) for each axial contour\n')

os.makedirs(OUTPUT_DIR,exist_ok=True)

# combines the up and down portions of a cell cross section
def combine_halves(c, n, midpoint=18, reduce=True):
    a = c[:,n,:]
    b = np.array(list(reversed(c[:,n-18, :])))
    res = np.vstack([a,b])
    if reduce:
        res = res[:,[0,2]]
    return res


# Loads points from a contour file
def fetch_pts(fname, dims=2):
    pts = open(fname,'r').readlines()
    pts = [_.strip().split(',') for _ in pts[5:]]
    return np.array([_[2:2+dims] for _ in pts],dtype=float)


# Calculates efc ratio
def efc_ratio(contour, order, doPlot):
    coeffs = elliptic_fourier_descriptors(contour, order=order)
    semiaxes = np.array([np.sqrt(coeffs[:,0]**2+coeffs[:,2]**2), np.sqrt(coeffs[:,1]**2+coeffs[:,3]**2)])
    efcr = 2*semiaxes[0][0]+2*semiaxes[1][0]+2
    efcr /= 2*np.sum(semiaxes[:,1:])
    
    if doPlot:
        dc = pyefd.calculate_dc_coefficients(contour)
        print(contour[0]-dc)

        recd = pyefd.reconstruct_contour(coeffs, locus=(0, 0), num_points=300)
        plt.plot(recd.T[0], recd.T[1])
        plt.scatter(contour.T[0]-dc[0], contour.T[1]-dc[1], color='k')
        plt.show()
        
#     print(np.array(semiaxes))
    return efcr, semiaxes


# Gets convex hull for a contour
def get_hull(c):
    hull = ConvexHull(c)
    hullpts = c[hull.vertices]
#     print('perimeter:', hull.area, hull.volume)

    return hull, hullpts


# calculates perimeter for a 2D contour
def get_perimeter(c):
    p = 0
    for a,b in zip(c,c[1:]):
        p+= np.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)
        
    p += np.sqrt((c[0][0]-c[-1][0])**2+(c[0][1]-c[-1][1])**2) # closure
    return p
        

# Gets area for a contour in 2D
def polyArea2D(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area


# calculates solidity, convexity, circularity in 2D
def get_metrics(c):
    _, hull = get_hull(c)
    
    area = polyArea2D(c)
    area_convex = polyArea2D(hull)
    
    perim = get_perimeter(c)
    perim_convex = get_perimeter(hull)
    
    solidity = area / area_convex
    convexity = perim_convex / perim
    formfactor = 3.14159*4*area/perim**2 # aka circularity
    
    return solidity, convexity, formfactor
    

def main():

    lines = ['parent', 'hgps', 'rs', 'huvec']

    for line in lines:
        dirs = list(sorted(glob.glob(os.path.join(BASE_DIR, f'lateral_{line}_P00_*/'))))
        files = list(sorted(glob.glob(os.path.join(dirs[0],'C_N*Z*.txt')))); print(len(files))
        print(len(dirs),files[0])
        
        all_line = []
        for celldir in dirs:
    #         print(celldir)
            cellfiles = list(sorted(glob.glob(os.path.join(celldir,'C_N*Z*.txt'))));
            
            contours_nuc = np.array([fetch_pts(f, dims=3) for f in cellfiles])
            
            all_cell = []
            for n in range(contours_nuc.shape[1]):
                combined = combine_halves(contours_nuc,n) 
                latmetrics = get_metrics(combined)
                all_cell.append(latmetrics)
    #             print(celldir, latmetrics)
            all_line.append(all_cell)
        all_line = np.array(all_line)
        print(np.shape(all_line))
        for cell, celldir in zip(range(len(all_line)), dirs):
            with open(os.path.join(OUTPUT_DIR,f'{line}_cell_{cell}.txt'),'w') as fout:
                fout.write('# '+celldir + '\n')
                for theta in all_line[cell]:
                    fout.write(','.join([str(round(_,4)) for _ in theta]) + '\n')


        with open(os.path.join(OUTPUT_DIR,f'xsection_summary.txt'),'a') as fout:
            to_write = [round(_,4) for _ in [np.mean(all_line[:,:,0]),np.mean(all_line[:,:,1]),np.mean(all_line[:,:,2])]]
            print(line, to_write)

            fout.write('# ' + line + '\n')

            meaned_cell = np.mean(all_line,1)
            print(meaned_cell.shape)

            for cell in meaned_cell:
                fout.write(','.join([str(round(_,4)) for _ in cell]) + '\n')
            fout.write('\n\n')

        np.save(os.path.join(OUTPUT_DIR,f'{line}.npy'),all_line)

            # print(np.mean(meaned_cell,0), np.std(meaned_cell,0))
            # fout.write(','.join([str(round(_,4)) for _ in np.mean(meaned_cell,0)]))

    summarize_metrics(OUTPUT_DIR, lines, 0)
    summarize_metrics(OUTPUT_DIR, lines, 1)
    summarize_metrics(OUTPUT_DIR, lines, 2)

def summarize_metrics(path, lines, n=0):
    line_data = [np.load(os.path.join(path, f'{line}.npy')) for line in lines]
    print('shape:', line_data[0].shape)

    with open(os.path.join(path, f'summary_{n}.txt'),'w') as fout:
        for k,line in enumerate(lines):
            temp = np.mean(line_data[k],1)
            # print('temp.shape', temp.shape)
            fout.write(f'{line}\t'+'\t'.join([str(round(t,4)) for t in temp[:,n]])+'\n')
        # print(','.join([str(np.mean(line_data[_],1)[n],4) for _ in range(len(line_data))]))


if __name__ == '__main__':
    main()
