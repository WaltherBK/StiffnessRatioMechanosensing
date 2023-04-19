# verify.py
# Compares contours output into two folders to confirm they are the same.

import glob
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

VERIFY_BASE = '/Users/asears/work/confocal/verification'

# parser = argparse.ArgumentParser(description='Build contours')
# parser.add_argument('dir1', type=str,
#                     help='first directory')
# parser.add_argument('dir2', type=str,
#                     help='second directory')
# parser.add_argument('name', type=str,
#                     help='name of comparison')
                
# args = parser.parse_args()
# argsd = vars(args)
# print('argsd:', argsd) # argsd: {'dir1': 'a', 'dir2': 'b', 'name': 'c'}

def verify(d1,d2,name=None, threshold=0):


    files1 = list(sorted(glob.glob(os.path.join(d1,'*Z*.txt'))))
    files2 = list(sorted(glob.glob(os.path.join(d2,'*Z*.txt'))))

    print(len(files1), files1[0])
    print(len(files2), files2[0])

    def fetch_pts(fname, dims=2):
        pts = open(fname,'r').readlines()
        pts = [_.strip().split(',') for _ in pts[5:]]
        return np.array([_[2:2+dims] for _ in pts],dtype=float)
            

    os.makedirs(VERIFY_BASE, exist_ok=True)

    if name is not None:
        verify_dir = os.path.join(VERIFY_BASE,name)
        os.makedirs(verify_dir, exist_ok=True)

    for f1, f2 in zip(files1,files2):

        stub1 = f1.split('/')[-1]

        p1 = fetch_pts(f1, dims=3)
        p2 = fetch_pts(f2, dims=3)

        if name is not None:
            plt.plot(p1.T[0], p1.T[1])
            plt.scatter(p2.T[0], p2.T[1])
            plt.title(stub1)
            plt.savefig(os.path.join(verify_dir, stub1+'.png'), bbox_inches='tight')
            plt.cla()

        diff = np.sum(np.abs(p1-p2))

        pct = diff / (np.sum(np.abs(p1+p2)/2.0))*100.0
        if pct > threshold:
            print(stub1, diff, pct)

if __name__ == '__main__':

    d1 = 'export_contours/gtest_hgps_P00_1/'
    d2 = 'export_contours/contours_hgps1_P00_1/'
    name = 'hgps1_1'

    verify(d1, d2, name)