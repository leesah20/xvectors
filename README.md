# X-vectors for speaker and gender embeddings

This is a PyTorch implementation for learning x-vectors that can be used as speaker or gender embeddings. The architecture follows what is described in 

D. Snyder, D. Garcia-Romero, D. Povey, and S. Khudanpur, “Deep neural network embeddings for text-independent speaker verification,” _in Proc. Interspeech_, 2017

with some changes made discussed in 

A. Kanagasundaram, S. Sridharan, G. Sriram, S. Prachi, and C. Fookes, “A study of x-vector based speaker recognition on short utterances,” _in Proc. Interspeech_, 2019.

For details on the experiments of this implementation, please see

Van Staden, Lisa. “Improving Unsupervised Acoustic Word Embeddings Using Segment- and Frame-Level Information.” Masters thesis, Stellenbosch University, 2021.

## Setup

### Data
The datasets for English and Xitsonga that was used for this implementation has to be downloaded separately. Please go to [ kamperh /
recipe_bucktsong_awe ](https://github.com/kamperh/recipe_bucktsong_awe) for details on where to download the datasets and how to extract the features.

### Docker
You can run this code inside a docker container. Build your image from Dockerfile.gpu or Dockerfile.cpu if you are using a GPU or not, respectively.

`docker build -f docker/&lt;DOCKER FILE NAME> -t &lt;IMAGE NAME>`

You'll have to mount the volumes containing your datasets when running the docker image. Update config/data_paths.json accordingly.

### Requirements

If you're not using docker, update config/data_paths.json to point to the paths of the datasets on your machine and install the following:
- Python 3.6 or higher
- torch
- scikit-learn
- numpy

## Run
The model configuration can be edited in config/mode_config.json.

To train the x-vectors as speaker embeddings on the english-full dataset, run the following command:

`python xvectors.py train config/model_config.json &lt;CHECKPOINT SAVE DIRECTORY>`

Note: remember to set the PYTHONPATH to the src folder.

For other run options see the list of argumants below.

| Arguments                                          | Description                                                                                                                                                                             |
|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| action (required)                                  | Choose from the following:<br/> -train (trains the model)<br/> -evaluate (runs evaluation on the model) <br/>-extract (saves x-vectors to a specified path) <br/>-all (runs all actions) |
| config_file (required)                             | Path to the config file e.g.  _config/model_config.json_                                                                                                                                |
| checkpoint_path (required)                         | Path to the directory where checkpoints will be saved or loaded.                                                                                                                        |
| --extract_language (default: english_full)         | The language dataset that should be used to extract the x-vectors from. Used only if all or extract is chosen for action.                                                               |
| --gender (optional)                                | A boolean flag to indicate that the x-vectors should be trained as gender embeddings.                                                                                                   |
| --language (default: english_full)                 | Choose the language on which the model should be trained.                                                                                                                               |
| --load-from-path (optional)                        | If specified, loads the weights from the given epoch before the model is trained e.g. --load-from-epoch 100                                                                             |
| --save_path (required if action is all or extract) | The path to which the npz containing the x-vectors should be saved.                                                                                                                     |
| --projection (default:lda)                         | The projection used when extracting the x-vectors. Choose from lda, pca or mean. Used only if all or extract chosen for action.|


