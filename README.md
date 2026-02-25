# ML-Deployment-using-FLASK
Create a House Price prediction model and deploy it as a Web App using Flask on a local system.

(Error encountered while trying to save the model with joblib module and retrieving the model with the pickle module. It was corrected by using the pickle module to save and load the model.)

Data set: House-Price.csv. We select only some of the numerical features.

regression-model.py includes the training of the Multiple Linear Regressor model.

index.html renders the 'GET' command and the user inputs the features to predict the house price.

result.html renders the 'POST' command and displays the prediction result.

script.py encapsulates the Flask app and its callbacks.
