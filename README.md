# Financial Forecasting ML
## Forecasting of the Forex market using machine learning techniques

This project tries to answer to 2 questions:

1.Can we predict whether a candle will be up or down ? ( Classification )

2.To which extent can we predict the exact daily closing price of a currency ? ( Regression )

This project used Support Vector Machines ( SVMs ) and Neural Networks.

**Python 2.7** is needed to run these files

*Methodology:*
* Gather daily EURUSD and USDJPY data from Metatrader in the form of csv files
* Clean the data (we only need Open, High, Close, Low data )
* Create input and output sets for classification and regression
* Create the mapping between the input/output sets for training and testing
* Train the models and test them
* Evaluate the models ( Accuracy for classification and Mean Square Error for regression )

## Sample outputs

*Output of FF_NN_24_12_1.py :*

![NN_Regression](https://github.com/IsmailAlaouiAbdellaoui/Financial-Forecasting-ML/blob/master/MSE-NN.JPG)
![NN_Regression2](https://github.com/IsmailAlaouiAbdellaoui/Financial-Forecasting-ML/blob/master/NN%20Regression-Actual-VS-Prediction(USDJPY).JPG)

*Output of NN_classification.py :*

![NN_Classification](https://github.com/IsmailAlaouiAbdellaoui/Financial-Forecasting-ML/blob/master/Accuracy-NN.JPG)

*Output of display_daily_eurusd_SVR.py :*

![SVM_Regression](https://github.com/IsmailAlaouiAbdellaoui/Financial-Forecasting-ML/blob/master/MSE-SVM.JPG)
![SVM_Regression2](https://github.com/IsmailAlaouiAbdellaoui/Financial-Forecasting-ML/blob/master/SVM%20Regression-Actual-VS-Prediction(EURUSD).JPG)

*Output of display_daily_eurusd_SVC.py :*

![SVM_Classification](https://github.com/IsmailAlaouiAbdellaoui/Financial-Forecasting-ML/blob/master/Accuracy-SVM.JPG)



## Results

For the **classification** :
* Simple SVM gave 51.72% of accuracy
* A 24*8*2*8*1 Neural Network gave of accuracy 52.37%

For the **Regression** :
* Simple SVM gave an MSE of 4.99e-05
* A 24*12*1 Neural Network gave an MSE of 0.7357

## Things to improve the models :
* Do feature selection using Chi Square , or RSE ( and compare them )
* Do 5 fold cross-validation using TimeSeriesSplit ( from sklearn )
* Do GridSearchCV to find the best parameters ( or use RandomSearchCV if too exhaustive )
* Explore how the structure of the NN affects accuracy/MSE ( in terms of depth and length of layers )
* Use Recurrent NNs
