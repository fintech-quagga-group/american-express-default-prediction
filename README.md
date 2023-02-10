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