import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from xml.dom import minidom
import tifffile

def main(root: str, out: str, nfr: bool) -> None:
    out_dirs = os.listdir(out)
    if 'results' in out_dirs:
        out_dirs.remove('results')
    dirs = [dir.split('_')[0] for dir in out_dirs]
    left_near = []
    for i, scene in enumerate(zip(dirs, out_dirs)):
        print(f'Processing Scene {i}')
        HH_root = os.path.join(root, scene[0], scene[1] + '_HH')
        HV_root = os.path.join(root, scene[0], scene[1] + '_HH')
        HH_raw = tifffile.imread(os.path.join(HH_root, 'imagery_HH.tif')).astype(float)
        HV_raw = tifffile.imread(os.path.join(HV_root, 'imagery_HV.tif')).astype(float)

        # read the calibration gains and the offset value for sigma_0 from the lutSigma file.
        # note that there are as many gain values as columns in the image, i.e. len(gains)==HH.shape[0].
        HH_sigma_cal_gains = minidom.parse(os.path.join(HH_root, 'lutSigma.xml')).getElementsByTagName('gains')
        HH_sigma_cal_offset = minidom.parse(os.path.join(HH_root, 'lutSigma.xml')).getElementsByTagName('offset')
        HH_gains = np.array([float(v) for v in HH_sigma_cal_gains[0].firstChild.nodeValue.split(' ')])
        HH_offset = float(HH_sigma_cal_offset[0].firstChild.nodeValue)

        HV_sigma_cal_gains = minidom.parse(os.path.join(HV_root, 'lutSigma.xml')).getElementsByTagName('gains')
        HV_sigma_cal_offset = minidom.parse(os.path.join(HV_root, 'lutSigma.xml')).getElementsByTagName('offset')
        HV_gains = np.array([float(v) for v in HV_sigma_cal_gains[0].firstChild.nodeValue.split(' ')])
        HV_offset = float(HV_sigma_cal_offset[0].firstChild.nodeValue)
        # obtain the calibrated backscatter values in absolute units
        hh = (HH_raw**2 + offset)/gains
        hv = (HV_raw**2 + offset)/gains

        # SAR data is usually transformed to the log domain (dB) for analysis. 
        # specify cutoff sigma_0 values for mapping pixel values to the range (0,1)
        SIGMA_MIN_HH_DB = -35 # dB
        SIGMA_MAX_HH_DB = 0   # dB
        SIGMA_MIN_HV_DB = -35 # dB
        SIGMA_MAX_HV_DB = 0   # dB
        max_thresh = 10**(SIGMA_MAX_HH_DB/10)
        min_thresh = 10**(SIGMA_MIN_HH_DB/10)

        hh[hh>max_thresh]=max_thresh
        hh[hh<min_thresh]=min_thresh

        hv[hv>max_thresh]=max_thresh
        hv[hv<min_thresh]=min_thresh
        # truncate the values outside the selected range and map to [0,1]
        hh_sigma_0 = (10*np.log10(hh) - SIGMA_MIN_HH_DB)/(SIGMA_MAX_HH_DB-SIGMA_MIN_HH_DB)
        hv_sigma_0 = (10*np.log10(hv) - SIGMA_MIN_HV_DB)/(SIGMA_MAX_HV_DB-SIGMA_MIN_HV_DB)

        tifffile.imsave(os.path.join(out, scene[1], 'imagery_HH_sigma0_4_by_4average.tif'), hh_sigma_0[::4,::4])
        tifffile.imsave(os.path.join(out, scene[1], 'imagery_HV_sigma0_4_by_4average.tif'), hv_sigma_0[::4,::4])

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Generate incidence '
                                                  'angle map.'))
    parser.add_argument('-r', '--root', type=str, dest='root',
                        default=('/home/major/data/61-scene'),
                        help='Root directory of dataset')
    parser.add_argument('-o', '--out', type=str, dest='out',
                        default=('/home/major/data/21-scene/resnet/multi_folder-2022'),
                        help='Directory of output')
    parser.add_argument('-b', action='store_true',
                        help='boolean for generateing near/far-range map')

    args = parser.parse_args()

    main(root=args.root, out=args.out, nfr=args.b)



