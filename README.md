# RAMP starting kit on Mars craters detection and classification

Authors: Frédéric Schmidt, Anthony Lagain, Joris van den Bossche & Alexandre Boucaud

[![Build Status](https://travis-ci.org/ramp-kits/mars_craters.svg?branch=master)](https://travis-ci.org/ramp-kits/mars_craters)

Go to [`ramp-worflow`](https://github.com/paris-saclay-cds/ramp-workflow) for more help on the [RAMP](http:www.ramp.studio) ecosystem.

After cloning, run

```
python download_data.py
```

the first time. It will create `data/imgs` and download the images there
using the names `<id>.png`, where `<id>`s are coming from `data/train.csv` and `data/test.csv`.


Install ramp-workflow (rampwf), then execute

```
ramp_test_submission
```

Get started on this RAMP with the [dedicated notebook](mars_craters_starting_kit.ipynb).
