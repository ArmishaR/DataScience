# DataScience
This repo is for group 15 of CAP 5771 Data Science course

## Exploratory Analysis
Within the analysis.py file you will be able to find the code used for the exploratory analysis. The results of this analysis is in the DS_Images folder. There are additional folders within DS_Images that correspond to the plots generated and the results of the three different correlation measures used. Various plots are grouped based on other variables, such as hour, month, site, or building type when applicable. The primary_use image is a bar chart of the building types of the data.

## Preprocessing

## Model Development
All scripts pertaining to model development can be accessed the *model_development* folder. There is a script for each model with the naming convention of **modelname_para_opt.py**. These scripts are used to test each model with different parameters to determine which results in the best performance of each model. Within each script, a 10-fold cross validation of the training data is conducted across each set of parameters within each model. 

## Model Evaluation
After which, the **model_dev.py** script is used to test the validation set with the parameters of each model that resulted in the best performance from the cross validation. 
