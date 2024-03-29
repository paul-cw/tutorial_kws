# README
This is a **tutorial** that lets you build your own **keyword spotter from scratch**. You can find a summary of the accompanying lecture in this [blog post](https://medium.com/@pcw_48167/speech-recognition-101-hands-on-keyword-spotting-1-3-29a757af5b3d).
We will learn about **audio data**, **neural networks** and streaming audio data.
Some of the libraries we will get to know, are:
- **librosa**: load and process audio files
- **pandas**: load and process dataframes
- **keras**: build, train and use neural network models

## usage on linux or mac machine
Create a virtual environment with:
- `python3 -m venv ./venvs/tutorial` with your systems python >= 3.8 installation (3.9 tested)
- activate the venv with `source ./venvs/tutorial/bin/activate`. (tutorial) should appear at the start of each line in your terminal

Alternatively you can use conda to create a conda environment with Python 3.9., via:
  - conda create -n tutorial python=3.9
  - conda activate tutorial

After that:
- install the dependencies listed in requirements.txt via `pip install -r requirements.txt`. This may take a while.
- if you are on a remote machine run `jupyter notebook --port=8888` and connect via putty or some other ssh client to open the notebook in your browser

## download the data
- to download the data manipulate the machine dependent entries in the config.py. The data_dir variable is used to determine where to download the data to
- run download_and_prepare_data.py (This is done from within exercise_01 as well)

## exercises
- start with the exercise notebooks
- the code
 should check for the dataset and download it, if it has not been done manually already.

## usage with google colab
 This is not tested carefully, therefore the method above is preferred.
 When using Colab anyway, you should use the ´requirements_colab´ file to install the requirements.
 - add this cell to a notebook: `!pip install -r tutorial_kws/requirements_colab.txt`

## preparations for the summer school
- run all the notebooks once, especially exercise_01 and exercise_02 a) and b). They prepare some data, this is quite costly and should be done in advance

## useful resources 

- [Deep Learning](https://www.deeplearningbook.org/)
- [FFT](https://www.youtube.com/watch?v=spUNpyF58BY)
- [Neural Networks Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=6,2&seed=0.40563&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=true&cosY=false&sinY=true&collectStats=false&problem=classification&initZero=false&hideText=false)
- [CNNs](https://setosa.io/ev/image-kernels/)
- [CNNs II](https://theano-pymc.readthedocs.io/en/latest/tutorial/conv_arithmetic.html)
