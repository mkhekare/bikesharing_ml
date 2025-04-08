# Bike Sharing Dataset Exploratory Analysis

This project provides an exploratory analysis of the Bike Sharing dataset from the UCI Machine Learning Repository. The focus is on the hourly data file, `bikes.csv`.

## Reference
Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg.

## Data Manipulation

```python
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
%matplotlib inline

# Setting parameters for visualization
params = {
    'legend.fontsize': 'x-large',
    'figure.figsize': (30, 10),
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large'
}

sn.set_style('whitegrid')
sn.set_context('talk')
plt.rcParams.update(params)
pd.options.display.max_colwidth = 600

# Load Dataset
hour_df = pd.read_csv('bikes.csv')
print("Shape of dataset::{}".format(hour_df.shape))
```

### Dataset Overview
The dataset contains 17 attributes and over 17,000 records. The attributes include:

- `instant`: Record ID
- `dteday`: Date
- `season`: Season (1: winter, 2: spring, 3: summer, 4: fall)
- `yr`: Year (0: 2011, 1: 2012)
- `mnth`: Month
- `hr`: Hour
- `holiday`: Whether the day is a holiday
- `weekday`: Day of the week
- `workingday`: Whether the day is a working day
- `weathersit`: Weather situation
- `temp`: Normalized temperature
- `atemp`: Normalized feeling temperature
- `hum`: Normalized humidity
- `windspeed`: Normalized wind speed
- `casual`: Count of casual users
- `registered`: Count of registered users
- `cnt`: Total count of users

### Data Types and Summary Stats
```python
# Data types of attributes
hour_df.dtypes
```

### Standardize Attribute Names
```python
hour_df.rename(columns={
    'instant': 'rec_id',
    'dteday': 'datetime',
    'holiday': 'is_holiday',
    'workingday': 'is_workingday',
    'weathersit': 'weather_condition',
    'hum': 'humidity',
    'mnth': 'month',
    'cnt': 'total_count',
    'hr': 'hour',
    'yr': 'year'
}, inplace=True)
```

### Typecast Attributes
```python
# Date time conversion
hour_df['datetime'] = pd.to_datetime(hour_df.datetime)

# Categorical variables
categorical_columns = ['season', 'is_holiday', 'weekday', 'weather_condition', 'is_workingday', 'month', 'year', 'hour']
for col in categorical_columns:
    hour_df[col] = hour_df[col].astype('category')
```

## Visualize Attributes, Trends, and Relationships

### Hourly Distribution of Total Counts
```python
fig, ax = plt.subplots()
sn.lineplot(data=hour_df, x='hour', y='total_count', hue='season', ax=ax)
ax.set(title="Season-wise hourly distribution of counts")
plt.show()
```

### Monthly Distribution of Total Counts
```python
fig, ax = plt.subplots()
sn.barplot(data=hour_df[['month', 'total_count']], x="month", y="total_count")
ax.set(title="Monthly distribution of counts")
plt.show()
```

### Correlation Analysis
```python
corrMatt = hour_df[["temp", "atemp", "humidity", "windspeed", "casual", "registered", "total_count"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
sn.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True)
plt.show()
```

## Conclusion
This analysis provides insights into the bike sharing trends based on various factors such as time of day, season, and weather. The findings can be used to improve bike sharing services and understand user behavior.

## Dependencies
- numpy
- pandas
- seaborn
- matplotlib
