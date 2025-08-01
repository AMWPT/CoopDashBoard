import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import seaborn as sns

#n = 534000  # 60% of 890000
#df = pd.read_csv("Lung cancer.csv").sample(frac=1, random_state=42).reset_index(drop=True).iloc[:n]



dfrand = pd.read_csv('health_lifestyle_classification.csv').sample(frac = 1, random_state=42).reset_index(drop=True)
n = int(len(dfrand) * 0.5)
df = dfrand.iloc[:n]

df = df.dropna(subset=["age", "gender", "bmi", "sleep_hours", "mental_health_score", "target"])


sns.boxplot(data=df, x='target', y='age')
plt.title("Age Distribution by Health Status")
plt.xlabel("Health Status")
plt.ylabel("Age")
plt.show()


sns.boxplot(data=df, x='target', y='cholesterol')
plt.title("Cholesterol by Health Status")
plt.show()

corr = df[['bmi', 'cholesterol', 'glucose', 'insulin', 'blood_pressure', 'heart_rate']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Between Health Metrics")
plt.show()


sns.scatterplot(data=df, x='physical_activity', y='bmi', hue='target')
plt.title("Physical Activity vs BMI (by Health Status)")
plt.show()


sns.scatterplot(data=df, x='screen_time', y='mental_health_score', hue='gender')
plt.title("Screen Time vs Mental Health (by Gender)")
plt.show()



plt.figure(figsize=(6,4))
df['gender'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
df['age'].hist(bins=30, color='lightgreen')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
df['bmi'].hist(bins=30, color='orange')
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Number of Individuals")
plt.tight_layout()
plt.show()

# Print BMI summary
print("Average BMI:", round(df['bmi'].mean(), 2))



plt.figure(figsize=(6,4))
plt.scatter(df['sleep_hours'], df['mental_health_score'], alpha=0.5)
plt.title("Sleep Hours vs Mental Health Score")
plt.xlabel("Sleep Hours")
plt.ylabel("Mental Health Score")
plt.tight_layout()
plt.show()



plt.figure(figsize=(6,4))
df['target'].value_counts().plot(kind='bar', color='purple')
plt.title("Target Distribution")
plt.xlabel("Health Status")
plt.ylabel("Count")
plt.tight_layout()
plt.show()



print("Total Participants:", len(df))
print("Avg Age:", round(df['age'].mean(), 1))
print("Avg Sleep Hours:", round(df['sleep_hours'].mean(), 2))
print("Avg Mental Health Score:", round(df['mental_health_score'].mean(), 2))
print("BMI Range:", round(df['bmi'].min(), 1), "-", round(df['bmi'].max(), 1))
