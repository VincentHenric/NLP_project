# IASD NLP project
Here is the repo of the NLP course project of Corentin Guérendel, Vincent Gouteux, Vincent Henric and Clara Simmat. It deals with the Quora duplicate questions problem.

## Project structure and files

The report of the project is the pdf file in the repo.

The code folder contains all the different experiments we realized for this project.

### Experiments
- `Benchmark_embeddings.ipynb` : Exhaustive study on the sentence embeddings
- `FuzzyWuzzy.ipynb` : Logistic regression and Gradient boosting on basic features & Fuzzy ratios.
- `MaLSTM_Siamese.ipynb` : Siamese network with LSTM and Manhattan distance (Tensorflow version)
- `MaLSTM_model.ipynb` : Siamese network with LSTM and Manhattan distance (Pytorch version and better results !)
- `SIamese_Net_Attention.ipynb` : Siamese network with attention layer (Tensorflow)
- `Sentence_Transformers_embbedings_bert.ipynb` : Retreiving bert embeddings and experiments on consine similarity
- `combine_features.ipynb` :	Combination of features from benchmark embedddings, and other features in a big model
- `compute_embeddings.py` : helper functions to compute sentence embeddings
- `modeling.py` : helper functions to launch some machine learning models
  
### Data management  

The three python files below are tools for cleaning and preparing the data (not used in every Jupyter Notebooks) :
- `cleaning.py` 
- `transform_dataset.py`
- `utils_cleaning.py`

### How to use

We run all the programs on Google Colab. GPU is highly recommended.
Many deep learning have been used, depending on how everyone in the project feel confortable with.
Some pretrained embeddings need to be downloaded. It should be clear in the notebooks.
