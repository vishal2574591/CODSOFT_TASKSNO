import pandas as pd
import numpy as np

print("=== TASK 1: Data Cleaning & Preprocessing ===\n")

# Load your dataset
df = pd.read_csv(r'C:\Users\BISHAL\Downloads\Teen_Mental_Health_Dataset (3).csv')

print("Original Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# 1. Handle missing values (if any)
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 2. Remove duplicates
df = df.drop_duplicates()

# 3. Fix data types (if needed)
# Already mostly good, but ensure numeric columns are correct
for col in ['age', 'daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep', 
            'academic_performance', 'physical_activity', 'stress_level', 'anxiety_level', 
            'addiction_level', 'depression_label']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Handle outliers (example for hours and scores)
df['daily_social_media_hours'] = df['daily_social_media_hours'].clip(0, 12)
df['sleep_hours'] = df['sleep_hours'].clip(3, 12)
df['screen_time_before_sleep'] = df['screen_time_before_sleep'].clip(0, 5)

# Create age group for better analysis
df['age_group'] = pd.cut(df['age'], bins=[12, 15, 18, 20], labels=['Early Teens', 'Mid Teens', 'Late Teens'])

# Save cleaned dataset
df.to_csv(r'C:\Users\BISHAL\Downloads\Teen_Mental_Health_Dataset (3).csv', index=False)

print("\n✅ Cleaning Completed Successfully!")
print("Cleaned Shape:", df.shape)
print("Cleaned file saved as: **cleaned_teen_mental_health.csv**")
print("\nPreview of Cleaned Data:")
print(df.head())
print("\nValue Counts for Depression Label:\n", df['depression_label'].value_counts())

import pandas as pd

print("=== Running Task 1: Creating Cleaned File ===")

# Load original dataset
df = pd.read_csv(r'C:\Users\BISHAL\Downloads\Teen_Mental_Health_Dataset (3).csv')

# Basic Cleaning
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df = df.drop_duplicates()

# Create age group
df['age_group'] = pd.cut(df['age'], bins=[12, 15, 18, 20], 
                         labels=['Early Teens', 'Mid Teens', 'Late Teens'])

# Save cleaned file
df.to_csv(r'C:\Users\BISHAL\Downloads\cleaned_teen_mental_health.csv', index=False)

print("✅ Cleaned file saved successfully!")
print("Shape:", df.shape)

# ====================== TASK 2: EXPLORATORY DATA ANALYSIS ======================

import matplotlib.pyplot as plt
import seaborn as sns

print("\n=== TASK 2: EXPLORATORY DATA ANALYSIS (EDA) ===\n")

# 1. Summary Statistics
print(df.describe())

# 2. Depression Label Analysis
print("\nDepression Label Distribution:")
print(df['depression_label'].value_counts())
print("\nIn Percentage:")
print(round(df['depression_label'].value_counts(normalize=True) * 100, 2))

# 3. Visualizations

# Age Group vs Depression
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='age_group', hue='depression_label', palette='Set2')
plt.title('Depression by Age Group')
plt.show()

# Social Media Hours vs Depression
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='depression_label', y='daily_social_media_hours', palette='Set3')
plt.title('Daily Social Media Hours by Depression Label')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

print("\n✅ Task 2 EDA Completed! Check the charts above.")

# ===================== TASK 3: DATA VISUALIZATION DASHBOARD =====================

print("\n=== TASK 3: DATA VISUALIZATION DASHBOARD ===\n")

import matplotlib
matplotlib.use('TkAgg')   # Force chart window to show

# 1. Pie Chart - Depression Distribution
print("Depression Distribution:")
print(round(df['depression_label'].value_counts(normalize=True) * 100, 2))

plt.figure(figsize=(7,7))
df['depression_label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'salmon'])
plt.title('Depression Label Distribution')
plt.ylabel('')
plt.show()

# 2. Bar Chart
plt.figure(figsize=(8,5))
sns.barplot(data=df, x='age_group', y='daily_social_media_hours', palette='viridis')
plt.title('Average Daily Social Media Hours by Age Group')
plt.ylabel('Hours')
plt.show()

print("\n✅ Charts should appear now. Close each chart to see the next one.")