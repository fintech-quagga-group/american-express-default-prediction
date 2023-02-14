# American Express Default Prediction - Kaggle Competition

In this project we develop a machine learning model that predicts credit defaults using real-world data from American Express to better manage risk in a consumer lending business. These models aim to provide a high-scoring solution to the Kaggle competition posted by American Express - [American Express Default Prediction](https://www.kaggle.com/c/amex-default-prediction). 

This repository holds and organizes our notebooks downloaded from Kaggle. The full list of links to each notebook is listed in the [Usage](#usage) section of this readme. Instructions on how to run the notebooks yourself are also included. 

Our report for our project will be listed in the [Project Report](#project-report) section, but a brief outline of our project and code is as follows: 

1. Collect, impute, encode, and preprocess the [compressed feather data](https://www.kaggle.com/datasets/munumbutt/amexfeather) and [aggregate data](https://www.kaggle.com/datasets/huseyincot/amex-agg-data-pickle) found from other competititors. 

    * Compressed Feather Data
        * [amex_generate_features](https://www.kaggle.com/code/ethansilvas/amex-generate-features)
        * [amex_generate_features_test](https://www.kaggle.com/ethansilvas/amex-generate-features-test)
    * Aggregate Data
        * [amex_impute_encode_agg_data](https://www.kaggle.com/ethansilvas/amex-impute-encode-agg-data)
2. Develop baseline models to see performance without much hyperparameter tuning or validation sets
    * All models in [Baseline Models](./Baseline_Models/) folder for training and predicting with the compressed and aggregate data sets
    * Examples: 
        * Logistic Regression - [baseline_logistic_regression_agg](https://www.kaggle.com/code/ethansilvas/baseline-logistic-regression-agg)
        * Shallow Neural Network - [baseline_shallow_nn_agg](https://www.kaggle.com/code/ethansilvas/baseline-shallow-nn)
3. Tune the best performing models using validation, hyperparameters, hidden layers, etc. to maximize M score
    * All models in [Tuned Models](./Tuned_Models/) folder for training and predicting with the aggregate data only, since we found that the aggregate data performed better for all models. 
    * Examples: 
        * Random Forest Classifier - [tuned_random_forest_classifier_agg](https://www.kaggle.com/code/ethansilvas/tuned-random-forest-classifier-agg)
        * Deep Neural Network - [tuned_DNN_agg](https://www.kaggle.com/ethansilvas/tuned-dnn-agg)

---

## Project Report 

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

    * Impute NaN values and one-hot encode the [compressed feather data](https://www.kaggle.com/datasets/munumbutt/amexfeather) and [aggregate data](https://www.kaggle.com/datasets/huseyincot/amex-agg-data-pickle)
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

### Data Collection 

The [original dataset](https://www.kaggle.com/competitions/amex-default-prediction/data?select=train_data.csv) posted by American Express had a few things to get around:  

* The training and testing sets were 16 and 33GB each 
* The columns, or features were hidden and grouped up to protect user privacy as so: 
    * D_* = Delinquency variables
    * S_* = Spend variables
    * P_* = Payment variables
    * B_* = Balance variables
    * R_* = Risk variables
* There were lots of NaN values and there were too many to be dropped from the dataset. 
* The data was already normalized 

Thankfully, other people in the competition came up with two datasets that we used in developing our models: 

* [The compressed feather data](https://www.kaggle.com/datasets/munumbutt/amexfeather)
* [The aggregate data](https://www.kaggle.com/datasets/huseyincot/amex-agg-data-pickle)

However, these datasets did not impute the NaN values or encode the categorical features, so we decided to create imputed and one-hot encoded versions of these datasets. We decided to impute the NaN valuse by replacing numerical columns with the mean for the column and categorical values with the most common value. We also opted to one-hot encode all of the categorical values. The code for this can be seen in our notebooks:

* Feather Data
    * [amex_generate_features](https://www.kaggle.com/code/ethansilvas/amex-generate-features)
    * [amex_generate_features_test](https://www.kaggle.com/ethansilvas/amex-generate-features-test)
* Aggregate Data 
    * [amex_impute_encode_agg_data](https://www.kaggle.com/ethansilvas/amex-impute-encode-agg-data)

Even with the compressed sized of these datasets, our TensorFlow models had trouble fitting so we followed the TensorFlow guides to building data pipelines: [Load a pandas DataFrame](https://www.tensorflow.org/tutorials/load_data/pandas_dataframe#full_example) and [Using tf.data with tf.keras](https://www.tensorflow.org/guide/data#using_tfdata_with_tfkeras). For our data, essentially all we needed to do was conver our columns to TensorFlow Input() objects and cast the datatypes into float32. This code can be seen at the beginning of each of our neural network files, ex: [baseline_shallow_nn_agg](https://www.kaggle.com/code/ethansilvas/baseline-shallow-nn)

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

To run our Kaggle notebooks you will need to: 

1. Sign in with your Kaggle account
2. Go to the notebook and click the "Copy & Edit" button at the top right<br><br>
    ![Kaggle copy and edit button](./Resources/Images/copy_and_edit.png)
3. Then you can either click the "Run All" button to run the file in the editor, or you can click the "Save Version" button that will give you the option to do a "Save and Run All (Commit)" which lets you run that version of the notebook in the background. Doing the save and run all option will likely be better since many of our notebooks take 30+ minutes to run in full. 

    ![Run all editor button](./Resources/Images/run_all.png)
    ![Save version button](./Resources/Images/save_version.png)


### Kaggle Links 

Note: Some files are split due to memory constraints.

* Data Collection
    * Feather Data
        * [amex_generate_features](https://www.kaggle.com/code/ethansilvas/amex-generate-features)
        * [amex_generate_features_test](https://www.kaggle.com/ethansilvas/amex-generate-features-test)
    * Aggregate Data 
        * [amex_impute_encode_agg_data](https://www.kaggle.com/ethansilvas/amex-impute-encode-agg-data)
* Baseline Models
    * Decision Tree Classifier 
        * [train_decision_tree_classifier](https://www.kaggle.com/code/ethansilvas/train-decision-tree-classifier)
        * [predict_decision_tree_classifier](https://www.kaggle.com/code/ethansilvas/predict-decision-tree-classifier)
    * Logistic Regression
        * [baseline_logistic_regression](https://www.kaggle.com/code/ethansilvas/baseline-logistic-regression)
    * Logistic Regression Aggregate 
        * [baseline_logistic_regression_agg](https://www.kaggle.com/code/ethansilvas/baseline-logistic-regression-agg)
    * Random Forest Classifier
        * [baseline_random_forest_classifier](https://www.kaggle.com/code/ethansilvas/baseline-random-forest-classifier)
    * Random Forest Classifier Aggregate
        * [baseline_random_forest_classifier_agg](https://www.kaggle.com/code/ethansilvas/baseline-random-forest-classifier-agg)
    * Shallow Neural Network 
        * [baseline_shallow_nn_agg](https://www.kaggle.com/code/ethansilvas/baseline-shallow-nn)
    * Shallow Neural Network Aggregate
        * [baseline_shallow_nn_agg](https://www.kaggle.com/code/ethansilvas/baseline-shallow-nn-agg)
* Tuned Models 
    * Shallow Neural Network
        * [tuned_shallow_nn_agg](https://www.kaggle.com/ethansilvas/tuned-shallow-nn-agg)
    * Deep Neural Network
        * [tuned_DNN_agg](https://www.kaggle.com/ethansilvas/tuned-dnn-agg)
    * Logistic Regression
        * [tuned_logistic_regression_agg](https://www.kaggle.com/code/ethansilvas/tuned-logistic-regression-agg)
    * Random Forest Classifier
        * [tuned_random_forest_classifier_agg](https://www.kaggle.com/code/ethansilvas/tuned-random-forest-classifier-agg)

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