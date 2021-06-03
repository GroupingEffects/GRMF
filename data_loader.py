import numpy as np
import glob
from PIL import Image
import scipy.io


# Yale faces
def create_faces(path, last):
    """

    :param path:
    :param last:
    :return:
    """
    all_pic = np.zeros((192, 168))
    all_names = glob.glob(path)
    for name in all_names[-last:]:
        tmp = Image.open(name)
        img = np.array(tmp)
        all_pic = np.concatenate((all_pic, img), axis=1)
    all_pic = all_pic[:, 168:]
    return all_pic


def load_main_datasets(datasets='YaleB'):  # type: (str) -> np.array
    """
    Select datasets, option includes: YaleB, JAFFE, ORL, COIL and ORL_small
    :param datasets:
    :return:
    """
    """load the 4 main datasets: YaleB, JAFFE, ORL, COIL"""
    # YaleB 192*168 with 210 pictures
    if datasets == 'YaleB':
        filtered_words = ['Ambient', '130E+20', '050E-40', '110E-20', '110E+65', '000E+90', '130E+20', '120E+00',
                          '110E+40', '110E+00', '120E+00', '095E+00', '110E+40', '110E+15', '110E-20', '030E+65']
        all_file_names = glob.glob('data/CroppedYale/*')[5:20]
        yaleb = np.zeros((192, 1))  # cut the first 1 ros
        for name in all_file_names:
            all_names = glob.glob(name + '/*.pgm')
            all_names = [sent for sent in all_names if not any(word in sent for word in filtered_words)]
            for i in range(0, 40, 3):
                tmp = np.array(Image.open(all_names[i]))[:192, :168]
                yaleb = np.concatenate((yaleb, tmp), axis=1)
        return yaleb[:, 1:]

    # ORL small 400 pictures with 32*32
    if datasets == 'ORL_small':
        mat = scipy.io.loadmat('data/ORL_32x32.mat')
        a = np.array(mat['fea'])
        orl = np.zeros((32, 1))
        for i in range(400):
            orl = np.concatenate((orl, a[i, :].reshape(32, 32).T), axis=1)
        return orl[:, 1:]

    # ORL origin, 200 pictures with 112*92
    if datasets == 'ORL':
        all_file_names = glob.glob('data/FaceDB_orl/*')
        orl = np.zeros((112, 1))  # cut the first 1 ros
        for name in all_file_names:
            all_names = glob.glob(name + '/*.png')
            for i in range(0, len(all_names), 2):
                tmp = np.array(Image.open(all_names[i]))
                orl = np.concatenate((orl, tmp), axis=1)
        return orl[:, 1:]

    # COIL-20 choose in every 7 picture, 206 pictures with 128*128
    if datasets == 'COIL':
        all_names = glob.glob('data/coil-20-proc/*.png')
        coil = np.zeros((128, 1))  # cut the first 1 ros
        for i in range(0, len(all_names), 7):
            tmp = np.array(Image.open(all_names[i]))
            coil = np.concatenate((coil, tmp), axis=1)
        return coil[:, 1:]

    # JAFFE, 213 pictures with size of 180*150
    if datasets == 'JAFFE':
        all_names = glob.glob('data/jaffe/*.tiff')
        jaffe = np.zeros((180, 1))  # cut the first 1 ros
        for name in all_names:
            tmp = np.array(Image.open(name))[50:230, 50:200]
            jaffe = np.concatenate((jaffe, tmp), axis=1)
        return jaffe[:, 1:]
