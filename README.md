# README
These is a tutorial that lets you build your own keyword spotter from scratch. You can find a summary of the accompanying lecture in this [blog post](https://medium.com/@pcw_48167/speech-recognition-101-hands-on-keyword-spotting-1-3-29a757af5b3d).

## Installation
Create a virtual environment with:
- `python3 -m venv ./venvs/tutorial` with your systems python >= 3.8 installation (3.9 tested)
- activate the venv with `source ./venvs/tutorial/bin/activate`. (tutorial) should appear at the start of each line in your terminal

Alternatively you can use conda to create a conda environment with Python 3.9., via:
  - conda create -n tutorial python=3.9
  - conda activate tutorial

After that:
- install the dependencies listed in requirements.txt via `pip install -r requirements.txt`. This may take a while.
- if you are on a remote machine run `jupyter notebook --port=8888` and connect via putty or some other ssh client to open the notebook in your browser


## Exercises
- to download the data manipulate the machine dependent entries in the config.py. The data_dir variable is used to determine where to download the data to
- run download_and_prepare_data.py 
- start with the exercises
