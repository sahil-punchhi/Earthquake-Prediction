# Earthquake-Prediction
To predict the time that an earthquake will occur in a laboratory test using Scikit-Learn (Pedregosa et al. (2011), XGBoost,CatBoost and LightGBM libraries for machine learning and support. The laboratory test applies shear forces to a sample of earth and rock containing a fault line. If the physics are ultimately shown to scale from the laboratory to the field, researchers will have the potential to improve earthquake hazard assessments that could save lives and billions of dollars in infrastructure. The metric used is Mean Absolute Error (MAE) and thus a lower value is better with zero representing a perfect fit.

## To install the dependencies:

Before running the actual code , we need to install all libraries used in the module using the following command:

```pip3 install -r requirements.txt ```

## To run the code 

Type the following command to run the code:

``` python3 main.py ``` 


Which will run the code in the 'main.py' file , applies the machine learning model depending on the function called under the 

```__name__=__main__:```
 
 section of the main.py file.


## Training Model Files
Our Implemention basically compares the accuracy of prediction of time to failure using different machine learning models , namely xgBoost, lightgbm and catBooster , which is implemented in the following files:

``` prediction_cbr.py  -> CatBoost ```
``` prediction_xgb.py  -> xgBoost ```
``` prediction_lgb.py  -> lightgbm ```


## Result Files

For each model, after the code runs to completion, four files are generated in the folder:
``` data_files/result/ ```
namely, ``` train_features.csv ``` which stores training data features, ``` test_features.csv ``` which stores test data features, ``` time_to_failure.csv ``` which stores actual time_to_failure values, and ``` sample_submission.csv ``` which stores predicted time_to_failure values.
