# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 16:58:38 2025

@author: Alyan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2.1: Data Processing
data = pd.read_csv("project1_data.csv") # Read in data
data = data.dropna().reset_index(drop=True) # Drop empty data rows

# 2.2: Data Visualization
fig1 = plt.figure()
plt.plot(data["Step"], data["X"], label="X")
plt.plot(data["Step"], data["Y"], label="Y")
plt.plot(data["Step"], data["Z"], label="Z")
plt.xlabel("Step")
plt.ylabel("Coordinate")
plt.legend()
plt.grid(True)
plt.show()

# 2.3: Correlation Analysis

# Create a heatmap of the absolute Pearson Correlation Matrix of the data
sns.heatmap(np.abs(data.corr(method='pearson')))

# 2.4: Classification Model Development/Engineering
