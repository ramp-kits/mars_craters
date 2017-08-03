import os
import shutil
import numpy as np
import pandas as pd
from subprocess import check_call
from skimage.io import imsave

N_QUAD = 140
URLBASE = 'sftp://DOMAIN_TO_SPECIFY/mars_craters/quadrangles/{:03d}.npz'


def download_file(url):
    print('Downloading {} ...'.format(url))
    check_call(['wget', url], shell=True)


def get_ids(df_train, df_test, index):
    ids_train = df_train[index]['id'].values
    ids_test = df_test[index]['id'].values
    local_ids_train = df_train[index]['quad_id'].values
    local_ids_test = df_test[index]['quad_id'].values
    absolute_ids = np.concatenate([ids_train, ids_test])
    local_ids = np.concatenate([local_ids_train, local_ids_test])

    return absolute_ids, local_ids


def save_quadrangle_to_png(filename, abs_ids, local_ids, out_dir):
    """
    Extract images as individual PNGs from compressed numpy file.

    Parameters
    ----------
    filename: str
        Path to numpy container for quadrangle data
    abs_ids: array of int
        Absolute index for the Mars dataset
    local_ids: array of int
        Array index within a quadrangle
    out_dir: str
        Path to output directory

    """
    data = np.load(filename)
    X_data = data[local_ids]

    print('Saving images in {}/<id>.png ...'.format(out_dir))
    for id_, img in zip(abs_ids, X_data):
        img_file = "{:06d}.png".format(id_)
        imsave(os.path.join(out_dir, img_file), img)


def main():
    urls = [URLBASE.format(id_quad) for id_quad in range(1, N_QUAD + 1)]
    filenames = [os.path.basename(url) for url in urls]

    img_dir = os.path.join('data', 'imgs')
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.mkdir(img_dir)

    df_train = pd.read_csv(os.path.join('data', 'train.csv'))
    df_test = pd.read_csv(os.path.join('data', 'test.csv'))

    for url, filename in zip(urls, filenames):
        if os.path.exists(filename):
            continue
        download_file(url)
        absolute_ids, local_ids = get_ids(df_train, df_test, index=filename)
        save_to_png(filename, absolute_ids, local_ids, img_dir)
        os.remove(filename)

    print('Images saved in {}/<id>.png ...'.format(img_dir))


if __name__ == '__main__':
    main()
