PROJECT TITLE ----- PREDICTIVE FAILURE ANALYSIS FOR PROACTIVE MAINTENANCE

This project focuses on early detection of abnormal degradation patterns in critical systems using predictive analytics. By estimating Remaining Useful Life (RUL) from multivariate sensor data, the system identifies high-risk operational behavior and supports proactive intervention.

DATASET

NASA Turbofan Engine Degradation Simulation Dataset

Multivariate time-series sensor data representing engine health over operational cycles

APPROACH

Cleaned and preprocessed sensor data

Engineered Remaining Useful Life (RUL) as a continuous risk indicator

Trained a Random Forest Regressor to model degradation behavior

Evaluated performance using MAE and RMSE

ANOMALY DETECTION EXTENSION

To enhance monitoring and security relevance:

1. Prediction errors were analyzed to detect abnormal behavior

2. A statistical threshold (mean + 2×std) was used to flag high-risk anomalies

3. Large deviations indicate potential abnormal system states

RESULTS
MAE: 29.63

RMSE: 41.45

Successfully detected rare but significant anomalous degradation patterns

TECHNOLOGIES 

Python, Matplotlib

Libraries: Pandas, NumPy, Scikit-learn

Steps to run the code:
Clone this repository to your local machine.

Install the required libraries: pip install pandas numpy scikit-learn

Place the train_FD001.txt data file inside a folder named data in the project directory.

Run the main script from your terminal: python main.py
