import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = 'US_Accidents_March23.csv'
df = pd.read_csv(file_path)


# Quick data check
print(df.info())
print(df.head())


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
# Figure setup
fig, axs = plt.subplots(4, 4, figsize=(24, 20))
axs = axs.flatten()


# 1. Top 10 States
accidents_by_state = df['State'].value_counts().head(10)
accidents_by_state.plot(kind='bar', color='skyblue', ax=axs[0])
axs[0].set_title('Top 10 States by Number of Accidents')

# 2. Accidents by Hour
accidents_by_hour = df['Hour'].value_counts().sort_index()
accidents_by_hour.plot(kind='line', marker='o', ax=axs[1])
axs[1].set_title('Accidents by Hour')
axs[1].set_xticks(range(0, 24))

# 3. Severity Distribution
severity_counts = df['Severity'].value_counts().sort_index()
severity_counts.plot(kind='bar', color='tomato', ax=axs[2])
axs[2].set_title('Severity Distribution')

# 4. Monthly Accidents
monthly_accidents = df.groupby('Month').size()
monthly_accidents.plot(kind='line', ax=axs[3])
axs[3].set_title('Monthly Accidents')

# 5. Top Cities
top_cities = df['City'].value_counts().head(10)
top_cities.plot(kind='bar', color='mediumseagreen', ax=axs[4])
axs[4].set_title('Top 10 Cities by Accidents')
axs[4].tick_params(axis='x', rotation=45)
axs[4].set_xlabel('')

# 6. Top Weather Conditions
weather_counts = df['Weather_Condition'].value_counts().head(10)
weather_counts.plot(kind='bar', color='steelblue', ax=axs[5])
axs[5].set_title('Top 10 Weather Conditions')
axs[5].tick_params(axis='x', rotation=45)

# 7. Day of Week
day_counts = df['Day_of_Week'].value_counts()
day_counts = day_counts.reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
day_counts.plot(kind='bar', color='orchid', ax=axs[6])
axs[6].set_title('Accidents by Day of Week')
axs[6].set_xlabel('')

# 8. Severity vs Temp Boxplot
sns.boxplot(x='Severity', y='Temperature(F)', data=df, ax=axs[7])
axs[7].set_title('Severity vs Temperature')

# 9. Severity by Weather Condition
severity_weather = df.groupby('Weather_Condition')['Severity'].mean().sort_values(ascending=False).head(15)
severity_weather.plot(kind='barh', ax=axs[8]) 
axs[8].set_title('Avg Severity by Weather Condition')
axs[8].set_xlabel('Average Severity')
axs[8].set_ylabel('Weather Condition')
axs[8].set_title('Avg Severity by Weather Condition')

# 10. Top Streets
top_streets = df['Street'].value_counts().head(10)
top_streets.plot(kind='bar', color='firebrick', ax=axs[9])
axs[9].set_title('Top 10 Streets', fontsize=10)
axs[9].tick_params(axis='x', rotation=45)

# 11. Visibility Bins
vis_counts = df['Visibility_Bin'].value_counts().sort_index()
vis_counts.plot(kind='bar', color='navy', ax=axs[10])
axs[10].set_title('Accidents by Visibility Range')
axs[10].tick_params(axis='x', rotation=45)
axs[10].set_xlabel('Visibility Range (mi)')

# 12. Wind Speed vs Severity
sns.scatterplot(x='Wind_Speed(mph)', y='Severity', data=df, alpha=0.3, ax=axs[11])
axs[11].set_title('Wind Speed vs Severity')

# 13. Duration Distribution (< 3hr)
df[df['Duration_Minutes'] < 180]['Duration_Minutes'].plot(kind='hist', bins=50, color='darkcyan', ax=axs[12])
axs[12].set_title('Accident Duration (< 3hr)')

# 14–15: Empty or filler
axs[13].axis('off')
axs[14].axis('off')
axs[15].axis('off')

for ax in axs:
    ax.tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels
    ax.tick_params(axis='y', labelsize=8)       # Optional: smaller font for y-axis

# For specific axes where label rotation is not needed (e.g., line plots or numeric x-axis), reset rotation
axs[1].tick_params(axis='x', labelrotation=0)  # Accidents by Hour
axs[3].tick_params(axis='x', labelrotation=0)  # Monthly Accidents
axs[7].tick_params(axis='x', labelrotation=0)  # Boxplot
axs[8].tick_params(axis='x', labelrotation=0)  # Horizontal bar
axs[11].tick_params(axis='x', labelrotation=0) # Scatter
axs[12].tick_params(axis='x', labelrotation=0) # Histogram


# Final layout
plt.tight_layout()
plt.suptitle('US Accidents Dashboard', fontsize=20, y=1.02)
plt.show()