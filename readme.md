# motivation

In my adventure to learn machine learning, I wanted to create a model that is able to determine/predict a yelp review rating from just text alone. In this process, I tried to make it as simple as possible to run. I created 4 different python files with separate tasks and meanings:

1) filter_data.py
2) create_model.py
3) predict_rating.py
4) lime_query_explanation.py

Run all these in order and you'll end up with a HTML page that shows the impact each word in a phase effects the models rating interpretation.

How To Set It Up:

<br>

## Import Github Repo:

    git clone https://github.com/Aschonn/ML-Predict-Ratings.git

<br>

## import dataset:

    https://www.yelp.com/dataset/download

    1) Enter Info and Click Download

    2) While Downloading Create a folder called 'dataset'

    3) Put Individual Yelp datasets inside dataset folder


## Create Virtualenv

#### install python3 and pip3 

    python3: https://phoenixnap.com/kb/how-to-install-python-3-ubuntu
    pip3: https://www.makeuseof.com/tag/install-pip-for-python/


#### Pip install virtualenv:

    python3 pip install virtualenv

#### Setup Up Virtualenv:

    $ which python3
    $ virtualenv -p <path_to_python3> env

#### Activate Env:

    $ source env/bin/activate

#### Install Dependencies:

    # pip3 install fasttext pathlib lime


## How To Run

On The Command Line Enter:

    1) python3 filter_data.py
    2) python3 create_model.py
    3) python3 predict_rating.py
    4) python3 lime_query_explanation.py

1) Will filter data so it is compatible with fasttext.

    fastText is a library for efficient learning of word representations and sentence classification.

2) Trains and tests the filtered data from before and create a model. 
3) Loads model and can predict rating from random sentence input.
4) Explains exactly what is happening. It displays an HTML page GUI that shows which words most greatly effected the models rating interpretation.

    https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.explanation
