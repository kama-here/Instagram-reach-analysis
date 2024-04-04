#instagram reach analysis using linear regression
#import required modules
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# data collection or reading of csv file and extraction data info and description from it
df= pd.read_csv("Instagram data.csv", encoding = 'latin1')
print(df.head())
df.info()
df.isnull().sum()
df.describe()
df.shape
# analyze the reach of instagram posts through graphs and plots
#distribution of impressions from home
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions from Home")
sns.distplot(df['From Home'])
plt.show()
#distribution of impressions from hashtags
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions from Hashtags")
sns.distplot(df['From Hashtags'])
plt.show()
#distribution of impressions from explore
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions from Explore")
sns.distplot(df['From Explore'])
plt.show()
#pie chart of impressions
home = df["From Home"].sum()
hashtags = df["From Hashtags"].sum()
explore = df["From Explore"].sum()
other = df["From Other"].sum()

labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
values = [home, hashtags, explore, other]
palette_color = sns.color_palette('pastel')

plt.pie(values, labels=labels, colors=palette_color, autopct='%.0f%%')
plt.show()
#relationship between likes and impressions
sns.lmplot(data=df, x="Impressions", y="Likes", fit_reg=True, scatter_kws={"s": 20}, line_kws={"lw":1})
ax = plt.gca()
ax.set_title("Relationship Between Likes and Impressions", pad=20)
plt.xlim(0, 15000)
plt.ylim(0, 500)
plt.show()
#relationship between comments and impressions
sns.lmplot(data=df, x="Impressions", y="Comments", fit_reg=True, scatter_kws={"s": 30}, line_kws={"lw":1})
ax = plt.gca()
ax.set_title("Relationship Between Comments and Total Impressions", pad=20)
plt.xlim(0, 15000)
plt.ylim(0, 25)
plt.show()
#relationship between shares and impressions
sns.lmplot(data=df, x="Impressions", y="Shares", fit_reg=True, scatter_kws={"s": 30}, line_kws={"lw":1})
ax = plt.gca()
ax.set_title("Relationship Between Shares and Impressions", pad=20)
plt.ylim(0, 80)
plt.xlim(0, 40000)
plt.show()
#relationship between saves and impressions
sns.lmplot(data=df, x="Impressions", y="Saves", fit_reg=True, scatter_kws={"s": 30}, line_kws={"lw":1})
ax = plt.gca()
ax.set_title("Relationship Between Post Saves and Total Impressions", pad=20)
plt.xlim(0, 15000)
plt.ylim(0, 700)
plt.show()
#correlating features to know which feature affect the most
df['Caption'] = pd.to_numeric(df['Caption'], errors='coerce')
df['Hashtags'] = pd.to_numeric(df['Hashtags'], errors='coerce')
correlation = df.corr()
print(correlation["Impressions"].sort_values(ascending=False))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
# analyzing conversion rate
conversion_rate = (df["Follows"].sum() / df["Profile Visits"].sum()) * 100
print(conversion_rate)
#relationship between profile visits and follows
sns.lmplot(data=df, x="Profile Visits", y="Follows", fit_reg=True, scatter_kws={"s": 30}, line_kws={"lw":1})
ax = plt.gca()
ax.set_title("Relationship Between Profile Visits and Followers Gained", pad=20)
# Select the highest correlated features for each feature
# You can adjust the threshold as needed
threshold = 0.8
selected_features = list()
for i in range(len(correlation.columns)):
    for j in range(i):
        if abs(correlation.iloc[i, j]) > threshold:
            colname = correlation.columns[i]
            selected_features.append(colname)

# Drop the features that are not among the highest correlated features
selected_data = df[selected_features]
print(selected_data)
transposed_data = selected_data.T
# Drop duplicate rows (which were columns in the original DataFrame)
transposed_data_unique = transposed_data.drop_duplicates()
# Transpose the DataFrame back to the original shape
data_unique = transposed_data_unique.T
print(data_unique)
print(data_unique.columns)
#split dependent and independent features
X=data_unique
Y=df['Impressions']
print(X)
print(Y)
#split train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=69)
model = LinearRegression()
model.fit(X_train, Y_train)
print("the accuracy score of the model is:",model.score(X_test, Y_test))
print(X.shape,X_train.shape,X_test.shape)
print(Y.shape,Y_train.shape,Y_test.shape)
features= np.array([[2500,1500,10,200,50]])
print("Instagram reach is:",model.predict(features))
y_pred = model.predict(X_test)
mae = mean_absolute_error(Y_test, y_pred)
print("mean absolute error is:",mae)
mse = mean_squared_error(Y_test, y_pred)
print("mean squared error is:",mse)
rmse = mean_squared_error(Y_test, y_pred, squared=False)
print("root mean squared error is:",rmse)
r2 = r2_score(Y_test, y_pred)
print("r2 score is:",r2)