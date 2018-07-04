from __future__ import print_function

import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

URLBASE = 'https://storage.ramp.studio/mars_craters_2/{}'
DATA = [
    'data_train.npy', 'data_test.npy']
LABELS = [
    'labels_train.csv', 'labels_test.csv']


def main(output_dir='data'):
    filenames = DATA + LABELS
    urls = [URLBASE.format(filename) for filename in filenames]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # notfound = []
    for url, filename in zip(urls, filenames):
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            continue

        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))


if __name__ == '__main__':
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()
