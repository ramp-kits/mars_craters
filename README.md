# RAMP starting kit on Mars craters detection and classification

Impact craters in planetary science are used to date and characterize planetary surfaces and study the geological history of planets. It is therefore an important task which traditionally has been achieved by means of visual inspection of images. The enormous number of craters, however, makes visual counting impractical. The challenge in this RAMP is to design an algorithm to automatically detect crater position and size based on satellite images.

Authors: Frédéric Schmidt, Anthony Lagain, Joris van den Bossche & Alexandre Boucaud

[![Build Status](https://travis-ci.org/ramp-kits/mars_craters.svg?branch=master)](https://travis-ci.org/ramp-kits/mars_craters)

First, make sure you have `ramp-workflow` installed:

```bash
$ pip install git+https://github.com/paris-saclay-cds/ramp-workflow
```

Go to [`ramp-workflow`](https://github.com/paris-saclay-cds/ramp-workflow) for more help on the [RAMP](http:www.ramp.studio) ecosystem.

Then download or ``git clone`` this repository and navigate to the directory:

```bash
$ git clone https://github.com/ramp-kits/mars_craters.git
$ cd mars_craters
```

Next, run the following command to download all data (around 1 Gb):

```bash
$ python download_data.py
```

This downloads the training and testing data into the `data` directory.

Further, make sure you have all dependencies installed. Eg using pip:

```bash
$ pip install -r requirements.txt
```

To check that everything is working, you can do a quick test of the starting kit submission:

```bash
$ ramp_test_submission --quick-test
```

To test an actual submission on the full data, you need to run:

```bash
$ ramp_test_submission --submission my_submission_name
```

Get started on this RAMP with the [dedicated notebook](mars_craters_starting_kit.ipynb).

### Amazon Machine Image (AMI)

We have built an AMI on the [Oregon site of AWS](https://us-west-2.console.aws.amazon.com). You can sign up and launch an instance following [this blog post](https://hackernoon.com/keras-with-gpu-on-amazon-ec2-a-step-by-step-instruction-4f90364e49ac). When asked for the AMI, search for `mars_craters_users`. Both `ramp-workflow` and this kit are pre-installed, along with the most popular deep learning libraries. We will use `g3.4xlarge` instances to train your models (in case you use deep learning). They cost about 1€/hour.
