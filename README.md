# DataScience
This repo is for group 15 of CAP 5771 Data Science course

## Exploratory Analysis
Within the analysis.py file you will be able to find the code used for the exploratory analysis. The results of this analysis is in the DS_Images folder. There are additional folders within DS_Images that correspond to the plots generated and the results of the three different correlation measures used. Various plots are grouped based on other variables, such as hour, month, site, or building type when applicable. The primary_use image is a bar chart of the building types of the data.

## Preprocessing

## Model Development
All scripts pertaining to model development can be accessed the *model_development* folder. There is a script for each model with the naming convention of **modelname_para_opt.py**. These scripts are used to test each model with different parameters to determine which results in the best performance of each model. Within each script, a 10-fold cross validation of the training data is conducted across each set of parameters within each model. 

## Model Evaluation
The results of the model evaluation are stored in the **evaluation/** folder. The results are broken into plots and text files.
The plots include:
- residual plots for every model
- prediction error plots for every model
- histograms of true vs. predicted values for every model (in both overlay and side-by-side comparison form)
There is a text file for each model type, plus the test data. Each text file includes:
- runtime to build the model
- runtime to get predictions
- root mean squared logarithmic error (RMSLE)
- R^2 score
- adjusted R^2 score
- mean absolute error (MAE)
The code used to generate the evaluations is found in the **model_development/** folder. The file **model_dev.py** can be used to generate plots and text files. The file **test_predictions.py** can be used to generate predictions for the test data.
