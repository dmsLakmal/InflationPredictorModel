
# import libraries

import pandas as pd

data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Nature Inspired Algorithm/CSV file/Inflation.csv')

# Data Preprocessing
print(data.columns)

print(data.head())

print(data.dtypes)

print(data.isnull().sum())

print(data.describe())

data

#save the clean data to the CSV file
data.to_csv('clean_data.csv', index=False)

# Recheck file CSV
import pandas as pd
data = pd.read_csv('/content/clean_data.csv')
data


data.describe()

# Data Mining

## Descriptive Analysis

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/content/clean_data.csv'
df = pd.read_csv(file_path)

# Inflaion Rates Of each countries
countries = df['Country']
inflation_rates = df['Inflation Rate (%)']
plt.figure(figsize=(14, 24))  
plt.barh(countries, inflation_rates, color='skyblue')
plt.xlabel('Inflation Rate (%)')
plt.ylabel('Country')
plt.title('Inflation Rates by Country')
plt.xscale('log') 
plt.grid(True, which="both", ls="--")
plt.show()

# Boxplot for identifying outliers in the inflation rates of different countries
inflation_rates = df['Inflation Rate (%)']
plt.figure(figsize=(10, 6))
plt.boxplot(inflation_rates, vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))
plt.xlabel('Inflation Rate (%)')
plt.title('Boxplot of Inflation Rates to Identify Outliers')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# correlation between inflation rate, GDP, and unemployment rate
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/content/clean_data.csv'
df = pd.read_csv(file_path)
# Display the first few rows of the dataframe to understand its structure
print(df.head())

selected_columns = df[['GDP (USD Trillions)', 'Unemployment Rate (%)', 'Inflation Rate (%)']]
correlation_matrix = selected_columns.corr()
print("Correlation Matrix:")
print(correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix between GDP, Unemployment Rate, and Inflation Rate')
plt.show()

# correlation between the inflation rate and GDP

# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/content/clean_data.csv'
df = pd.read_csv(file_path)
print(df.head())
selected_columns = df[['GDP (USD Trillions)', 'Inflation Rate (%)']]
correlation_matrix = selected_columns.corr()
print("Correlation Matrix between GDP and Inflation Rate:")
print(correlation_matrix)
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between GDP and Inflation Rate')
plt.show()

# Correlation between the inflation rate and unemployment rate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
file_path = '/content/clean_data.csv'
df = pd.read_csv(file_path)
print(df.head())
inflation_unemployment = df[['Inflation Rate (%)', 'Unemployment Rate (%)']]
correlation = inflation_unemployment.corr().iloc[0, 1]
print("Correlation between Inflation Rate and Unemployment Rate:")
print(correlation)
plt.figure(figsize=(10, 6))
sns.regplot(x='Unemployment Rate (%)', y='Inflation Rate (%)', data=inflation_unemployment)
plt.title('Correlation between Inflation Rate and Unemployment Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Inflation Rate (%)')
plt.show()

# GDP (USD Trillions) by Country

import geopandas as gpd
# Load world map data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.rename(columns={'name': 'Country'})
world_gdp = world.merge(df[['Country', 'GDP (USD Trillions)']], on='Country', how='left')
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world_gdp.boundary.plot(ax=ax, linewidth=1)
world_gdp.plot(column='GDP (USD Trillions)', ax=ax, legend=True,
               legend_kwds={'label': "GDP (USD Trillions)",
                            'orientation': "vertical"},  # Set orientation to vertical
               cmap='YlGnBu', missing_kwds={'color': 'lightgrey'})
plt.title('GDP (USD Trillions) by Country')
plt.axis('off')

plt.show()

# Unemployment Rate (%) by Countryimport geopandas as gpd"""
import geopandas as gpd
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.rename(columns={'name': 'Country'})
world_unemployment = world.merge(df[['Country', 'Unemployment Rate (%)']], on='Country', how='left')
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world_unemployment.boundary.plot(ax=ax, linewidth=1)
world_unemployment.plot(column='Unemployment Rate (%)', ax=ax, legend=True,
                        legend_kwds={'label': "Unemployment Rate (%)",
                                     'orientation': "vertical"},
                        cmap='OrRd', missing_kwds={'color': 'lightgrey'})
plt.title('Unemployment Rate (%) by Country')
plt.axis('off')

plt.show()

# Predective Analysis (Model Traning)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load the dataset
data = pd.read_csv('/content/clean_data.csv')

# Prepare the data
X = data[['GDP (USD Trillions)', 'Unemployment Rate (%)']]
y = data['Inflation Rate (%)']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the neural network model
model = Sequential()
model.add(Dense(32, input_dim=X_scaled.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Number of times to train the model
num_epochs = 10

for epoch in range(num_epochs):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    loss = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Epoch {epoch+1} - Test Loss: {loss} - R^2 Score: {r2}')

# Save the model and scaler to disk
model.save('model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)