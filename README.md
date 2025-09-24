PROJECT TITLE ----- PREDICTIVE FAILURE ANALYSIS FOR PROACTIVE MAINTENANCE

A brief description of the project goes here. Explain the problem you are solving and the overall goal.

Key Features and Methodology 🛠️
Data Sourcing: The dataset used for this project was the NASA Turbofan Engine Degradation Simulation Dataset.

Data Preparation: The raw data was cleaned by removing irrelevant and constant columns. New features, such as the Remaining Useful Life (RUL), were engineered for the model.

Modeling: A Random Forest Regressor was built to predict the RUL of the engines. The model was trained on 80% of the data and evaluated on the remaining 20%.

Performance Evaluation: The model's performance was measured using standard regression metrics.

Results and Outcomes 📈
The model achieved a Mean Absolute Error (MAE) of 29.63 and a Root Mean Squared Error (RMSE) of 41.45.

This means the model's predictions for the RUL were, on average, off by about 30 cycles, which is a strong result for this dataset.

Technologies Used 💻
Language: Python

Libraries: Pandas, NumPy, Scikit-learn

How to Run the Code 🚀
Clone this repository to your local machine.

Install the required libraries: pip install pandas numpy scikit-learn

Place the train_FD001.txt data file inside a folder named data in the project directory.

Run the main script from your terminal: python main.py
