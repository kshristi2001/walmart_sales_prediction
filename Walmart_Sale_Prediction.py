#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[2]:


#load the data into dataframe
df = pd.read_csv('Walmart.csv')


# In[3]:


#print the data
df.head()


# In[4]:


#print the shape of the dataset
df.shape


# In[5]:


#copying the dataset
original_df = df.copy(deep=True)


# In[6]:


# Convert the 'Date' column in the DataFrame 'df' to datetime format
df['Date'] = pd.to_datetime(df['Date'])


# In[7]:


df.head()


# In[8]:


#adding new columns to the dataset
df['weekday'] = df.Date.dt.weekday
df['month'] = df.Date.dt.month
df['year'] = df.Date.dt.year


# In[9]:


df.head()


# In[10]:


#printing the information of the dataset
df.info()


# In[11]:


#Checking the descriptive statistics of the dataframe
df.describe()


# In[12]:


#Displaying the null values
df.isnull().sum()


# In[13]:


# Count unique values in each column of the DataFrame 'df'
df.nunique().sort_values()


# In[14]:


# Plot the distribution of 'Weekly_Sales' using a histogram and display the plot.
plt.figure(figsize=[8,4])
sns.distplot(df['Weekly_Sales'], color='b',hist_kws=dict(edgecolor="black", linewidth=2), bins=30)
plt.title('Distribution of Weekly Sales')
plt.show()


# In[15]:


# Display the heatmap of the correlation matrix
plt.figure(figsize=[25,20])
sns.heatmap(df.corr(), annot=True,vmin=-1, vmax=1, center=0, cmap='BuGn')
plt.title('Correlation Matrix')
plt.show()


# In[16]:


# Plot the total 'Weekly_Sales' over time as a line chart, grouping the sales by 'Date' to show trends.
plt.figure(figsize=[8,4])
df.groupby('Date')['Weekly_Sales'].sum().plot(kind='line', title='Weekly Sales Over Time')
plt.show()


# In[17]:


# Plot the total 'Weekly_Sales' by each 'Store' as a bar chart
plt.figure(figsize=[14,7]) 
df.groupby('Store')['Weekly_Sales'].sum().plot(kind='bar', title='Weekly sales by store') 
plt.show()


# In[18]:


# Plot the average 'Weekly_Sales' for holiday and non-holiday weeks using a bar chart to compare sales impact.
plt.figure(figsize = [7,4])
df.groupby('Holiday_Flag')['Weekly_Sales'].mean().plot(kind='bar', title='Average Weekly Sales on Holidays vs. Non-Holidays')
plt.show()


# In[19]:


# Create a scatter plot to explore the relationship between 'Temperature' and 'Weekly_Sales'
df.plot(kind='scatter', x='Temperature', y='Weekly_Sales', title='Weekly Sales vs. Temperature')
plt.show()


# In[20]:


# Aggregate 'Weekly_Sales' by 'year' and 'month'
monthly_sales = df.groupby(['year', 'month'])['Weekly_Sales'].sum().unstack()

# Plot the data
monthly_sales.plot(kind='line', figsize=(12, 8))

# Set the title and labels
plt.title('Monthly Sales Over Years')
plt.xlabel('Year')
plt.ylabel('Total Sales')

# Adjust the x-axis ticks to show only the years
plt.xticks(ticks=monthly_sales.index, labels=monthly_sales.index)

# Show the plot
plt.show()


# In[22]:


#copying the data
df3 = df.copy()


# In[23]:


# Convert 'Holiday_Flag' into a dummy variable
df3 = pd.get_dummies(df3, drop_first=True, columns=['Holiday_Flag'], prefix = 'Holiday_Flag')
df3.rename(columns={'Holiday_Flag_1': 'Holiday_Flag'}, inplace=True)
print(df3)

df3.columns


# In[24]:


#dropping the date feature
df3.drop(['Date'], inplace = True, axis = 1)
print(df3)


# In[25]:


#concatenate the modified DataFrame with new dummy variables
df3 = pd.concat([df3.drop(['year','Weekly_Sales','Temperature','Fuel_Price','CPI','Unemployment','Holiday_Flag','weekday','month','Store'], axis=1), 
                 pd.DataFrame(pd.get_dummies(df3, columns=['year','weekday','month','Store'], drop_first=True))],axis=1)


# In[28]:


#Removal of outlier:
df1 = df3.copy()

features = ['Unemployment', 'Fuel_Price', 'CPI', 'Temperature']

for i in features:
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[df1[i] <= (Q3+(1.5*IQR))]
    df1 = df1[df1[i] >= (Q1-(1.5*IQR))]
    df1 = df1.reset_index(drop=True)
display(df1.head())


# In[29]:


df = df1.copy()


# In[30]:


#Splitting the data intro training & testing sets
X = df.drop('Weekly_Sales',axis=1)
Y = df['Weekly_Sales']
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
Train_X.reset_index(drop=True,inplace=True)


# In[31]:


# Initialize the StandardScaler to standardize features
std = StandardScaler()

# Fit the scaler to the training data and transform it
print('\033[1mStandardardization on Training set'.center(120))
Train_X_std = std.fit_transform(Train_X)
Train_X_std = pd.DataFrame(Train_X_std, columns=X.columns)
display(Train_X_std.describe())

# Transform the testing data using the already fitted scaler
print('\n','\033[1mStandardardization on Testing set'.center(120))
Test_X_std = std.transform(Test_X)
Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns)
display(Test_X_std.describe())


# In[34]:


# Initialize and fit a PCA model to the standardized training data
pca = PCA().fit(Train_X_std)


# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Determine the number of components to explain 95% of the variance
k = np.where(cumulative_variance >= 0.95)[0][0] + 1 

print(f"Number of components to explain 95% of variance: {k}")


# In[35]:


# Assuming k is the number of components determined from cumulative variance
pca = PCA(n_components=k)

# Fit and transform training data
Train_X_std_pca = pca.fit_transform(Train_X_std)  
Test_X_std_pca = pca.transform(Test_X_std) 


# In[36]:


# Initialize a Linear Regression model
LR = LinearRegression()
LR.fit(Train_X_std_pca, Train_Y)

# Predict the training set and test set outcomes using the fitted model
pred_train = LR.predict(Train_X_std_pca)
pred_test = LR.predict(Test_X_std_pca)


# In[37]:


# Calculate metrics for the training set
train_r2 = r2_score(Train_Y, pred_train)
train_mse = mean_squared_error(Train_Y, pred_train)
train_rmse = np.sqrt(train_mse)

# Calculate metrics for the testing set
test_r2 = r2_score(Test_Y, pred_test)
test_mse = mean_squared_error(Test_Y, pred_test)
test_rmse = np.sqrt(test_mse)

# Create a DataFrame to hold the evaluation metrics
evaluation_metrics = pd.DataFrame({
    'Metric': ['R2 Score', 'MSE', 'RMSE'],
    'Training Set': [train_r2, train_mse, train_rmse],
    'Testing Set': [test_r2, test_mse, test_rmse]
})

# Display the DataFrame
print("Model Evaluation Metrics:")
display(evaluation_metrics)


# In[38]:


# Plot a scatter graph of actual vs. predicted values for the test set.
plt.figure(figsize=(10, 6))
plt.scatter(Test_Y, pred_test, color='blue', label='Actual vs Prediction')
plt.plot([Test_Y.min(), Test_Y.max()], [Test_Y.min(), Test_Y.max()], 'r--', label='Ideal Fit')
plt.title('Test vs Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


# In[39]:


# Plotting training data
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.scatter(Train_X_std_pca[:, 0], Train_Y, color='blue', label='Actual', alpha=0.5)
plt.scatter(Train_X_std_pca[:, 0], pred_train, color='orange', label='Prediction', alpha=0.5)
plt.title('Training Data: Actual vs Prediction')
plt.legend()

# Plotting testing data
plt.subplot(1, 2, 2)
plt.scatter(Test_X_std_pca[:, 0], Test_Y, color='blue', label='Actual', alpha=0.5)
plt.scatter(Test_X_std_pca[:, 0], pred_test, color='orange', label='Prediction', alpha=0.5)
plt.title('Testing Data: Actual vs Prediction')
plt.legend()

plt.show()


# In[47]:


# Set the degree of the Polynomial
degree = 2 

# Create a pipeline that creates polynomial features and then fits a linear regression mode
poly_reg_model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('feature_selection', SelectKBest(f_regression, k=50)),  # Select the top 50 features
    ('linear', LinearRegression())
])


# Fit the model
poly_reg_model.fit(Train_X_std_pca, Train_Y)


# Make predictions
pred_train = poly_reg_model.predict(Train_X_std_pca)
pred_test = poly_reg_model.predict(Test_X_std_pca)

# Calculate metrics for the training set
train_r2 = r2_score(Train_Y, pred_train)
train_mse = mean_squared_error(Train_Y, pred_train)
train_rmse = np.sqrt(train_mse)

# Calculate metrics for the testing set
test_r2 = r2_score(Test_Y, pred_test)
test_mse = mean_squared_error(Test_Y, pred_test)
test_rmse = np.sqrt(test_mse)

# Create a DataFrame to hold the evaluation metrics
evaluation_metrics = pd.DataFrame({
    'Metric': ['R2 Score', 'MSE', 'RMSE'],
    'Training Set': [train_r2, train_mse, train_rmse],
    'Testing Set': [test_r2, test_mse, test_rmse]
})

# Display the DataFrame
print("Model Evaluation Metrics:")
display(evaluation_metrics)


# In[48]:


# Visualizing the Test vs Prediction for Polynomial Regression
plt.figure(figsize=(10, 6))
plt.scatter(Test_Y, pred_test, color='blue', label='Actual vs Prediction')
plt.plot([Test_Y.min(), Test_Y.max()], [Test_Y.min(), Test_Y.max()], 'r--', label='Ideal Fit')
plt.title('Test vs Prediction - Polynomial Regression')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()


# In[49]:


# Visual comparison of actual and predicted values in polynomial regression in training data
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.scatter(Train_X_std_pca[:, 0], Train_Y, color='blue', label='Actual', alpha=0.5)
plt.scatter(Train_X_std_pca[:, 0], pred_train, color='orange', label='Prediction', alpha=0.3)
plt.title('Training Data: Actual vs Prediction - Polynomial Regression')
plt.legend()


# Visual comparison of actual and predicted values in polynomial regression in test data
plt.subplot(1, 2, 2)
plt.scatter(Test_X_std_pca[:, 0], Test_Y, color='blue', label='Actual', alpha=0.5)
plt.scatter(Test_X_std_pca[:, 0], pred_test, color='orange', label='Prediction', alpha=0.3)
plt.title('Testing Data: Actual vs Prediction - Polynomial Regression')
plt.legend()

plt.show()


# In[ ]:




