import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
months = pd.date_range(start='2020-01-01', periods=60, freq='ME')
sales = np.random.randint(1000, 5000, size=len(months))
data = pd.DataFrame({'Month': months, 'Sales': sales})

data.set_index('Month', inplace=True)

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12)
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

plt.figure(figsize=(12, 6))
plt.plot(train, label='Training Data')
plt.plot(test, label='Testing Data')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('Sales Forecast')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()
