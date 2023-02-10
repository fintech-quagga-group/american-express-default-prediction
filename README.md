# American Express Default Prediction - Kaggle

### Outline

[American Express Default Prediction](https://www.kaggle.com/c/amex-default-prediction)

**Competition Description**<br><br>
Whether out at a restaurant or buying tickets to a concert, modern life counts on the convenience of a credit card to make daily purchases. It saves us from carrying large amounts of cash and also can advance a full purchase that can be paid over time. How do card issuers know we’ll pay back what we charge? That’s a complex problem with many existing solutions—and even more potential improvements, to be explored in this competition.
Credit default prediction is central to managing risk in a consumer lending business. Credit default prediction allows lenders to optimize lending decisions, which leads to a better customer experience and sound business economics. Current models exist to help manage risk. But it's possible to create better models that can outperform those currently in use.
American Express is a globally integrated payments company. The largest payment card issuer in the world, they provide customers with access to products, insights, and experiences that enrich lives and build business success.
In this competition, you’ll apply your machine learning skills to predict credit default. Specifically, you will leverage an industrial scale data set to build a machine learning model that challenges the current model in production. Training, validation, and testing datasets include time-series behavioral data and anonymized customer profile information. You're free to explore any technique to create the most powerful model, from creating features to using the data in a more organic way within a model.
If successful, you'll help create a better customer experience for cardholders by making it easier to be approved for a credit card. Top solutions could challenge the credit default prediction model used by the world's largest payment card issuer—earning you cash prizes, the opportunity to interview with American Express, and potentially a rewarding new career.

**Process**
1. Data collection
    * Impute and one-hot encode data
    * Aggergate data
2. Baseline models
    * LGBM
    * XGBoost
    * Scikit-learn models
        * Randomforestclassifier
        * SVC
        * KNN classifier
    * TensorFlow neural network (very simple; 1 hidden layer)
3. Tune models
    * Different hyperparameters
    * More hidden layers and different activation functions
4. Determine best performing models

### Data Used

### Summary

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