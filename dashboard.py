import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

file_path = 'US_Accidents_March23.csv'
df = pd.read_csv(file_path, nrows=200000)

plt.rcParams.update({
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 7,
    'figure.titlesize': 10
})

# Ensure datetime conversion
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
df['Hour'] = df['Start_Time'].dt.hour
df['Month'] = df['Start_Time'].dt.to_period('M')
df['Day_of_Week'] = df['Start_Time'].dt.day_name()
df['Duration_Minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
df['Visibility_Bin'] = pd.cut(df['Visibility(mi)'], bins=[0, 1, 3, 5, 10, 20, 50], include_lowest=True)
df = df.dropna(subset=['Start_Time'])
df = df.dropna(subset=['End_Time'])

plt.figure(figsize=(16, 12))

# 1. Top 10 States
plt.subplot(4, 4, 1)
accidents_by_state = df['State'].value_counts().head(10)
accidents_by_state.plot(kind='bar', color='skyblue')
plt.title('Top 10 States by Number of Accidents')

# 2. Accidents by Hour
plt.subplot(4, 4, 2)
accidents_by_hour = df['Hour'].value_counts().sort_index()
accidents_by_hour.plot(kind='line', marker='o')
plt.title('Accidents by Hour')
plt.xticks(range(0, 24), fontsize=6)

# 3. Severity Distribution
plt.subplot(4, 4, 3)
severity_counts = df['Severity'].value_counts().sort_index()
severity_counts.plot(kind='bar', color='tomato')
plt.title('Severity Distribution')

# 4. Monthly Accidents
plt.subplot(4, 4, 4)
monthly_accidents = df.groupby('Month').size()
monthly_accidents.plot(kind='line')
plt.title('Monthly Accidents')

# 5. Top Cities
plt.subplot(4, 4, 5)
top_cities = df['City'].value_counts().head(10)
top_cities.plot(kind='bar', color='mediumseagreen')
plt.title('Top 10 Cities by Accidents')
plt.xticks(rotation=45, fontsize=6)
plt.xlabel('')

# 6. Top Weather Conditions
plt.subplot(4, 4, 6)
weather_counts = df['Weather_Condition'].value_counts().head(10)
weather_counts.plot(kind='bar', color='steelblue')
plt.title('Top 10 Weather Conditions')
plt.xticks(rotation=45, fontsize=6)

# 7. Day of Week
plt.subplot(4, 4, 7)
day_counts = df['Day_of_Week'].value_counts()
day_counts = day_counts.reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
day_counts.plot(kind='bar', color='orchid')
plt.title('Accidents by Day of Week')
plt.xlabel('')

# 8. Severity vs Temp Boxplot
plt.subplot(4, 4, 8)
sns.boxplot(x='Severity', y='Temperature(F)', data=df)
plt.title('Severity vs Temperature')

# 9. Severity by Weather Condition
plt.subplot(4, 4, 13)
severity_weather = df.groupby('Weather_Condition')['Severity'].mean().sort_values(ascending=False).head(15)
severity_weather.plot(kind='barh')
plt.title('Avg Severity by Weather Condition')
plt.xlabel('Average Severity')
plt.ylabel('Weather Condition')
plt.tick_params(axis='y', labelsize=6)

# 10. Top Streets
plt.subplot(4, 4, 14)
top_streets = df['Street'].value_counts().head(10)
top_streets.plot(kind='bar', color='firebrick')
plt.title('Top 10 Streets')
plt.xticks(rotation=45, fontsize=6)

# 11. Prediction Graph
plt.subplot(4, 4, 15)
top_weather = df['Weather_Condition'].value_counts().head(3).index
monthly_periods = df['Month'].sort_values().unique()
month_labels = [str(m) for m in monthly_periods]
last_period = monthly_periods[-1]
for i in range(1, 7):
    next_period = (last_period + i)
    month_labels.append(str(next_period))

colors = ['orange', 'green', 'red']
for idx, weather in enumerate(top_weather):
    weather_df = df[df['Weather_Condition'] == weather]
    monthly_accidents = weather_df.groupby('Month').size()
    monthly_accidents = monthly_accidents.reindex(monthly_periods, fill_value=0)
    months_numeric = np.arange(len(monthly_accidents)).reshape(-1, 1)
    accidents_values = monthly_accidents.values

    model = LinearRegression()
    model.fit(months_numeric, accidents_values)
    future_months = np.arange(len(monthly_accidents), len(monthly_accidents) + 6).reshape(-1, 1)
    future_preds = model.predict(future_months)
    all_months = np.concatenate([months_numeric, future_months])
    all_accidents = np.concatenate([accidents_values, future_preds])

    plt.plot(all_months.flatten(), all_accidents, label=f'{weather} (pred)', color=colors[idx], linestyle='--')
    plt.plot(months_numeric.flatten(), accidents_values, label=f'{weather} (actual)', color=colors[idx])

plt.title('Predicted Monthly Accidents by Weather (Next 6 Months)')
step = 4
month_labels_short = [pd.Period(m).strftime('%b') if hasattr(pd, 'Period') else str(m)[5:8] for m in month_labels]
xticks = list(range(0, len(month_labels_short), step))
xticklabels = [month_labels_short[i] if i in xticks else '' for i in range(len(month_labels_short))]
plt.xticks(range(len(month_labels_short)), xticklabels, rotation=45, fontsize=6, ha='right')
plt.tick_params(axis='x', labelsize=6, pad=1)
plt.tick_params(axis='y', labelsize=6, pad=1)
plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

plt.tight_layout(pad=1.0)
plt.subplots_adjust(wspace=0.3, hspace=0.6, right=0.85)
plt.suptitle('US Accidents Dashboard', fontsize=16, y=1.03)
plt.show()
