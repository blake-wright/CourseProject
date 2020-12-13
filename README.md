# Classification Competition - Analyzing Twitter Tweets

## Setting up your environment

You will need the following libraries to successfully run my project:

Library        Version Used    Pip install cmd
--------------------------------------------------------
Tensorflow     2.3.1           pip install tensorflow
Sklearn-learn  0.23            pip install sklearn
Transformers   3.5.1           pip install transformers
Pandas         1.1.3           pip install pandas
Numpy          1.18.5          pip install numpy
Torch          1.7.1           pip install torch


The following videos can be used as a reference on how to setup a miniconda python environment if you
don't have any of the libraries and want some setup automatically. However, for the tensorflow.yml file
you will want to update the '''tensorflow=2.0''' to '''tensorflow=2.3.1'''. Or just copy the below.

'''
name: tensorflow

dependencies:
    - python=3.7
    - pip>=19.0
    - jupyter
    - tensorflow=2.0
    - scikit-learn
    - scipy
    - pandas
    - pandas-datareader
    - matplotlib
    - pillow
    - tqdm
    - requests
    - h5py
    - pyyaml
    - flask
    - boto3
    - pip:
        - bayesian-optimization
        - gym
        - kaggle
'''

For Windows:
https://www.youtube.com/watch?v=RgO8BBNGB8w

For MacOS:
https://www.youtube.com/watch?v=MpUvdLD932c&t=372s
