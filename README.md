# RAMP starting kit on Mars craters detection and classification

[![Build Status](https://travis-ci.org/ramp-kits/mars_craters.svg?branch=master)](https://travis-ci.org/ramp-kits/mars_craters)

_Authors: Joris van den Bossche, Alexandre Boucaud, Frédéric Schmidt & Anthony Lagain_

Impact craters in planetary science are used to date and characterize planetary surfaces and study the geological history of planets. It is therefore an important task which traditionally has been achieved by means of visual inspection of images. The enormous number of craters, however, makes visual counting impractical. The challenge in this RAMP is to design an algorithm to automatically detect crater position and size based on satellite images.

#### Set up

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
  ```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook](mars_craters_starting_kit.ipynb).

#### Amazon Machine Image (AMI)
 
We have built an AMI on the [Oregon site of AWS](https://us-west-2.console.aws.amazon.com). You can sign up and launch an instance following [this blog post](https://hackernoon.com/keras-with-gpu-on-amazon-ec2-a-step-by-step-instruction-4f90364e49ac). When asked for the AMI, search for `mars_craters_2_users`. Both `ramp-workflow` and this kit are pre-installed, along with the most popular deep learning libraries. We will use `p3.2xlarge` instances to train your models. They cost about 3€/hour. Alternativaly you can also use `p2.xlarge` instances which cost 1€/hour and 3-4x slower than `p3.2xlarge`.


#### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](http:www.ramp.studio) ecosystem.




