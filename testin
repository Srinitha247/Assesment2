import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load Dataset (Replace with your dataset)
df = pd.read_csv("screen_time.csv")

# Data Cleaning
# Handling missing values
df.dropna(inplace=True)  # Simple approach; customize as needed

# Data Preprocessing (if needed)
# Convert categorical variables to numerical (if applicable)
df = pd.get_dummies(df, drop_first=True)

# Statistical Moments
mean = df.mean()
variance = df.var()
skewness = df.skew()
kurtosis = df.kurtosis()

# Display statistical moments
print("Mean:\n", mean)
print("Variance:\n", variance)
print("Skewness:\n", skewness)
print("Kurtosis:\n", kurtosis)

# Relational Plot (Scatter Plot Example)
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1])
plt.title("Relational Plot")
plt.show()

# Categorical Plot (Bar Chart Example)
df.iloc[:, 0].value_counts().plot(kind='bar')
plt.title("Categorical Plot")
plt.show()

# Statistical Plot (Correlation Heatmap)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Clustering (K-Means)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df.iloc[:, :3])  # Using first 3 numerical columns
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['Cluster'])
plt.title("K-Means Clustering")
plt.show()

# Regression Model (Linear Regression)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Regression Performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
