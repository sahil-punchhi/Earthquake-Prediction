# Earthquake-Prediction
To predict the time that an earthquake will occur in a laboratory test using Scikit-Learn (Pedregosa et al. (2011), XGBoost (Chen &amp; Guestrin, 2016) and LightGBM (Ke, et al., 2017) libraries for machine learning and support. The laboratory test applies shear forces to a sample of earth and rock containing a fault line. If the physics are ultimately shown to scale from the laboratory to the field, researchers will have the potential to improve earthquake hazard assessments that could save lives and billions of dollars in infrastructure. The metric used is Mean Absolute Error (MAE) and thus a lower value is better with zero representing a perfect fit.

## Training Model Files
Our Implemention basically compares the accuracy of prediction of time to failure using different machine learning models , namely xgBoost, lightgbm and catBooster , which is implemented in the following files:

``` prediction_cbr.py  -> CatBoost
 prediction_xgb.py  -> xgBoost
 prediction_lgb.py  -> lightgbm ```


## Feature generation Files

