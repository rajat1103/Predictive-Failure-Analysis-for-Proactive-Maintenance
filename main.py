import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Define all 26 column names and load the dataset
columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
for i in range(1, 22):
    columns.append(f'sensor_measurement_{i}')

# Add the final two columns that are causing the data leak
columns.append('RUL_leak')
columns.append('Label_leak')

df = pd.read_csv('data/train_FD001.txt', sep=' ', header=None, names=columns)

# Step 2: Clean the data
# Now we explicitly drop the two columns that are leaking the answer
df.drop(columns=['RUL_leak', 'Label_leak'], inplace=True)

# Remove any other empty or constant columns
df.dropna(axis=1, how='all', inplace=True)
std_dev = df.std()
constant_columns = std_dev[std_dev == 0].index
df.drop(columns=constant_columns, inplace=True)

# Step 3: Create the target variable (RUL)
# This time we are confident it is being calculated correctly
rul_df = pd.DataFrame(df.groupby('unit_number')['time_in_cycles'].max()).reset_index()
rul_df.columns = ['unit_number', 'max_time_in_cycles']
df = df.merge(rul_df, on=['unit_number'], how='left')
df['RUL'] = df['max_time_in_cycles'] - df['time_in_cycles']
df.drop(columns=['max_time_in_cycles'], inplace=True)

# Step 4: Normalize the sensor data
sensor_cols = [col for col in df.columns if 'sensor_measurement' in str(col)]
scaler = MinMaxScaler()
df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

# Step 5: Split the data for modeling
feature_cols = [col for col in df.columns if 'op_setting' in str(col) or 'sensor_measurement' in str(col)]
X = df[feature_cols]
y = df['RUL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build the machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 7: Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the final results
print('\n--- Model Performance ---')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')