import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from xml.dom import minidom


def main(root: str, out: str, nfr: bool) -> None:
    out_dirs = os.listdir(out)
    if 'results' in out_dirs:
        out_dirs.remove('results')
    dirs = [dir.split('_')[0] for dir in out_dirs]
    left_near = []
    for scene in zip(dirs, out_dirs):
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
        plt.xlabel('Range Direction')
        plt.ylabel('Incidence Angle')
        if nfr:
            plt.savefig(f'{out}/{scene[1]}incifence_angle.png')
        if incidence_angle[0] < incidence_angle[-1]:
            left_near.append(1)  #left is near range
        else:
            left_near.append(0)  #right is near range
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
