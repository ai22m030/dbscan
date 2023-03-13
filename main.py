import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from comparer import plot_clusters

# Load the datasets
possum = np.loadtxt("possum.csv", delimiter=',', dtype=str, skiprows=1)

y_possum = possum[:, 3]
X_possum = possum[:, [5, 6]].astype('float')
le_possum = preprocessing.LabelEncoder().fit(y_possum)
y_possum_scaled = le_possum.transform(y_possum)

walmart = np.loadtxt("Walmart.csv", delimiter=',', dtype=str, skiprows=1)

y_walmart = walmart[:, 0].astype('float')
X_walmart = walmart[:, [2, 5]].astype('float')

weather = np.loadtxt("weatherHistory.csv", delimiter=',', dtype=str, skiprows=1)

y_weather = weather[:, 11]
X_weather = weather[:, [3, 6]].astype('float')
le_weather = preprocessing.LabelEncoder().fit(y_weather)
y_weather_scaled = le_weather.transform(y_weather)

# Print plots
X_train, X_test, y_train, y_test = train_test_split(X_possum, y_possum_scaled, test_size=0.2, random_state=42)
plot_clusters(X_train, X_test, y_train, y_test, 'Possum dataset')

X_train, X_test, y_train, y_test = train_test_split(X_walmart, y_walmart, test_size=0.2, random_state=42)
plot_clusters(X_train, X_test, y_train, y_test, 'Walmart dataset')

X_train, X_test, y_train, y_test = train_test_split(X_weather, y_weather_scaled, test_size=0.2, random_state=42)
plot_clusters(X_train, X_test, y_train, y_test, 'Weather dataset')

