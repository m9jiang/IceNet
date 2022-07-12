import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from xml.dom import minidom


def main(root: str, out: str, nfr: bool) -> None:
    our_dirs = os.listdir(out)
    dirs = [dir.split('_')[0] for dir in our_dirs]
    for scene in zip(dirs, our_dirs):
        # use beta_0 to sigma_0 ratio to get the incidence angle
        sigma_cal_gains = minidom.parse(
            os.path.join(root, scene[0], scene[1] + '_HH', 'lutSigma.xml')
        ).getElementsByTagName('gains')
        gains = np.array(
            [float(v) for v in sigma_cal_gains[0].firstChild.nodeValue.split(' ')]
        )
        beta_cal_gains = minidom.parse(
            os.path.join(root, scene[0], scene[1] + '_HH', 'lutBeta.xml')
        ).getElementsByTagName('gains')
        bgains = np.array(
            [float(v) for v in beta_cal_gains[0].firstChild.nodeValue.split(' ')]
        )
        incidence_angle = np.arcsin(bgains/gains)*180/np.pi
        plt.plot(incidence_angle)
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Generate incidence '
                                                  'angle map.'))
    parser.add_argument('-r', '--root', type=str, dest='root',
                        default=('D:\\Data\\For_Max'),
                        help='Root directory of dataset')
    parser.add_argument('-o', '--out', type=str, dest='out',
                        default=('D:\\Data\\Resnet\\Multi_folder-2022'),
                        help='Directory of output')
    parser.add_argument('-b', action='store_true',
                        help='boolean for generateing near/far-range map')

    args = parser.parse_args()

    main(root=args.root, out=args.out, nfr=args.b)
