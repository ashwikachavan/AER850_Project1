# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 13:29:02 2025

@author: Ashwika
"""

# Step 1
import pandas as pd

# Step 1: Read data from CSV file
file_path = "Project 1 Data.csv"  # Replace with your file path if needed
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
# print(df)

# Step 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV file
file_path = "Project 1 Data.csv"  # Replace with your file path if needed
df = pd.read_csv(file_path)

# Plot histograms using NumPy
plt.figure(figsize=(10, 6))

# Histogram for X
plt.subplot(3, 1, 1)
counts, bins = np.histogram(df['X'], bins=30)
plt.hist(bins[:-1], bins, weights=counts, edgecolor='black')
plt.title('Histogram of X')
plt.xlabel('X')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Histogram for Y
plt.subplot(3, 1, 2)
counts, bins = np.histogram(df['Y'], bins=30)
plt.hist(bins[:-1], bins, weights=counts, edgecolor='black')
plt.title('Histogram of Y')
plt.xlabel('Y')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Histogram for Z
plt.subplot(3, 1, 3)
counts, bins = np.histogram(df['Z'], bins=30)
plt.hist(bins[:-1], bins, weights=counts, edgecolor='black')
plt.title('Histogram of Z')
plt.xlabel('Z')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 3: Correlation Analysis

# Read the CSV file
file_path = "Project 1 Data.csv"  # Replace with your path if needed
df = pd.read_csv(file_path)

# Compute Pearson correlation matrix
correlation_matrix = df.corr(method='pearson')

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Pearson Correlation Heatmap")
plt.show()

# Pairplot to visualize feature relationships
sns.pairplot(df, hue="Step", diag_kind="hist", palette="tab10")
plt.suptitle("Pairplot of Features Colored by Step", y=1.02)
plt.show()

# Print correlation of each feature with the target variable (Step)
feature_target_corr = correlation_matrix['Step'].drop('Step')
print("Correlation of features with Step:\n")
print(feature_target_corr.sort_values(ascending=False))
