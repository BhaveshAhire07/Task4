import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Define file path for US Accidents dataset
file_path = 'D:\\Bhavesh goldi\\Python file\\US_Accidents_March23.csv'

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: The file {file_path} does not exist. Please download 'US_Accidents_March23.csv' from "
          f"https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents and save it to {file_path}")
    sys.exit(1)

try:
    # Load the dataset with a sample limit for testing (remove nrows for full dataset)
    df = pd.read_csv(file_path, low_memory=False, nrows=100000)  # Adjust or remove nrows for full data
    print(f"Dataset loaded successfully. Shape: {df.shape} (rows, columns)")

    # Data Cleaning and Preparation
    print("\nInitial Data Info:")
    print(df.info())

    print("\nMissing Values (Top 10 Columns):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    # Convert Start_Time to datetime and extract hour
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Hour'] = df['Start_Time'].dt.hour

    # Fill missing values for key columns
    df['Weather_Condition'] = df['Weather_Condition'].fillna('Unknown')
    df['Severity'] = df['Severity'].fillna(df['Severity'].median())

    # Create Road_Condition based on available road features
    road_features = ['Crossing', 'Junction', 'Bump', 'Traffic_Calming', 'Railway', 'Roundabout', 'Station', 'Stop', 'Amenity']
    available_road_features = [col for col in road_features if col in df.columns]
    df['Road_Condition'] = df[available_road_features].any(axis=1).map({True: 'Has Road Feature', False: 'Normal Road'}).astype('category')

    # Identify Patterns
    accidents_by_hour = df['Hour'].value_counts().sort_index()
    accidents_by_weather = df['Weather_Condition'].value_counts().head(10)
    accidents_by_road = df['Road_Condition'].value_counts()

    print("\nPattern 1: Accidents by Hour of Day:")
    print(accidents_by_hour.head())
    print("\nPattern 2: Accidents by Weather Condition (Top 10):")
    print(accidents_by_weather)
    print("\nPattern 3: Accidents by Road Condition:")
    print(accidents_by_road)

    # Contributing Factors
    severity_by_weather = df.groupby('Weather_Condition')['Severity'].mean().sort_values(ascending=False).head(10)
    severity_by_road = df.groupby('Road_Condition')['Severity'].mean()
    print("\nContributing Factors: Average Severity by Weather Condition (Top 10):")
    print(severity_by_weather)
    print("\nContributing Factors: Average Severity by Road Condition:")
    print(severity_by_road)

    # Visualizations
    plt.figure(figsize=(12, 6))
    accidents_by_hour.plot(kind='bar', color='skyblue')
    plt.title('Accidents by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Accidents')
    plt.xticks(rotation=0)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()
    plt.savefig('accidents_by_hour.png')
    print("Saved 'accidents_by_hour.png'")

    plt.figure(figsize=(12, 6))
    accidents_by_weather.plot(kind='bar', color='lightgreen')
    plt.title('Accidents by Weather Condition (Top 10)')
    plt.xlabel('Weather Condition')
    plt.ylabel('Number of Accidents')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig('accidents_by_weather.png')
    print("Saved 'accidents_by_weather.png'")

    plt.figure(figsize=(8, 6))
    accidents_by_road.plot(kind='bar', color='orange')
    plt.title('Accidents by Road Condition')
    plt.xlabel('Road Condition')
    plt.ylabel('Number of Accidents')
    plt.tight_layout()
    plt.show()
    plt.savefig('accidents_by_road.png')
    print("Saved 'accidents_by_road.png'")

    sample_df = df.sample(n=min(10000, len(df)), random_state=42)
    plt.figure(figsize=(12, 8))
    plt.scatter(sample_df['Start_Lng'], sample_df['Start_Lat'], alpha=0.5, s=1, color='red')
    plt.title('Accident Hotspots (Geographic Locations - Sampled)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(label='Accident Density (Sampled)')
    plt.tight_layout()
    plt.show()
    plt.savefig('accident_hotspots.png')
    print("Saved 'accident_hotspots.png'")

    plt.figure(figsize=(12, 6))
    severity_by_weather.plot(kind='bar', color='purple')
    plt.title('Contributing Factors: Average Severity by Weather Condition (Top 10)')
    plt.xlabel('Weather Condition')
    plt.ylabel('Average Severity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig('severity_by_weather.png')
    print("Saved 'severity_by_weather.png'")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)

# Ensure all plots are closed
plt.close('all')
print("Analysis complete. Check the directory for saved PNG files if plots didn't display.")