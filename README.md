# American Express Default Prediction - Kaggle

In this project we develop a machine learning model that predicts credit defaults using real-world data from American Express to better manage risk in a consumer lending business. These models aim to provide a high-scoring solution to the Kaggle competition posted by American Express - [American Express Default Prediction](https://www.kaggle.com/c/amex-default-prediction). 

## Outline

### Competition Description

This is a binary classification problem where our objective is to predict the probability that a customer does not pay back their credit card balance amount in the future based on their monthly customer profile. We are required to develop a model that generates predictions on the provided test set data and submit the predictions in a .csv file that is scored using the competition's evaluation metric `M` which is defined as: 

`The evaluation metric, 𝑀, for this competition is the mean of two measures of rank ordering: Normalized Gini Coefficient, 𝐺, and default rate captured at 4%, 𝐷.`

The highest possible score of `M` is a 1.0 and the top leaderboard score is 0.80977. 

### Project Goals 

In developing a solution for this competition we hope to: 
* Impute and encode our own dataset
* Use new models like XGBoost and LGBM
* Test techniques like dropout and regularization for neural networks 
* Score as close to 0.80 as possible

### Process

1. Data collection
    * Impute NaN values and one-hot encode the [compressed data](https://www.kaggle.com/datasets/munumbutt/amexfeather) and [aggregate data](https://www.kaggle.com/datasets/huseyincot/amex-agg-data-pickle)
2. Develop baseline models
    * LGBM
    * XGBoost
    * Scikit-learn models
        * LogisticRegression
        * DecisionTreeClassifier
        * RandomForestClassifier
        * SVC
        * KNN Classifier
    * Shallow neural network (1 hidden layer)
3. Tune models
    * Different hyperparameters specific to each model 
    * More hidden layers and different activation functions
4. Determine best performing models

### Data Collection 



## Summary

---

## Technologies

This is a Python 3.7 project ran using a JupyterLab in a conda dev environment. 

The following dependencies are used: 
1. [Jupyter](https://jupyter.org/) - Running code 
2. [Conda](https://github.com/conda/conda) (4.13.0) - Dev environment
3. [Pandas](https://github.com/pandas-dev/pandas) (1.3.5) - Data analysis
4. [Matplotlib](https://github.com/matplotlib/matplotlib) (3.5.1) - Data visualization
5. [Numpy](https://numpy.org/) (1.21.5) - Data calculations + Pandas support
6. [hvPlot](https://hvplot.holoviz.org/index.html) (0.8.1) - Interactive Pandas plots 
7. [holoviews](https://holoviews.org/) (1.15.2) - Interactive Pandas plots

---

## Installation Guide

If you would like to run the program in JupyterLab, install the [Anaconda](https://www.anaconda.com/products/distribution) distribution and run `jupyter lab` in a conda dev environment.

To ensure that your notebook runs properly you can use the [requirements.txt](/Resources/requirements.txt) file to create an exact copy of the conda dev environment used to create the notebook. 

Create a copy of the conda dev environment with `conda create --name myenv --file requirements.txt`

Then install the requirements with `conda install --name myenv --file requirements.txt`

---

## Usage

The Jupyter notebook []() will provide all steps of the data collection, preparation, and analysis. Data visualizations are shown inline and accompanying analysis responses are provided.

Our presentation slides for this project are in the Resources folder: []()

---

## Contributors

[Ethan Silvas](https://github.com/ethansilvas) <br>
[Naomy Velasco](https://github.com/naomynaomy) <br>
[Karim Bouzina](https://github.com/karim985) <br>
[Jeff Crabill](https://github.com/jeffreycrabill) <br>

---

## License

This project uses the [GNU General Public License](https://choosealicense.com/licenses/gpl-3.0/)