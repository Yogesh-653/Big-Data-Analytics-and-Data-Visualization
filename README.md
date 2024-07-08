                                                                  Project Title:
                                                  Productivity Prediction of Garment Employees
                                Analytics and predicting the model by using PySpark and visualization by Tableau

                                
Abstract:
This paper analyses the productivity of employee who is working in garment industry based on the various factors by accurately predicting productivity and identify potential areas for improvement to enhance overall efficiency and profitability by using machine learning classification and regression algorithm Four method used in this paper: Linear regression, Decision Tress, Random forest and Gradient Boosting.
Data analysis and ML was applied using PySpark. Then Visualization done using Tableau following python libraries are pandas, seaborn and matplotlib used. Out of four algorithm Gradient Boosting performed very well and gave more accuracy.
Keyword—Machine Learning, Data Analysis,PySpark, Linear regression, Decision Tress, Random forest and Gradient Boosting.
1.Introduction:

This report was the module on big data management and data visualization. The project focuses on big data it identifies and chooses the best analytical methods for the Big Data analysis and discussion of the findings. The main aim is to predict our model by using the PySpark with comes under the Apache Spark application under the python programming language by using some machine learning methods the finding the best method and their best accuracy level and overall charts present in the tableau was displayed in this report. 
Task Performed in this project:
1.Loading the dataset and pre-processing by using PySpark library 
2. EDA process for overall dataset
3.Using the appropriate classification and regression machine learning algorithm	
4. visualization of data using tableau
5.Finding of result and suggestion

2.Dataset description:
About Dataset: 
The dataset to be used for this project contains historical data from garment manufacturing units, including information about garment employees and the factors affecting their productivity. The dataset which contains missing values and their own attributes.

Context of Dataset:
	The Garment Industry is one of the key examples of the industrial globalization of this modern day. It is a highly labour-intensive industry with lots of manual processes. Satisfying the huge global demand for garment products is mostly dependent on the production and delivery performance of the employees in the garment manufacturing companies. So it is highly needed among the garment industry to track, analyse and predict the productivity performance of the worker in working teams in their factories. 

Dataset Attribute:
	This dataset includes important attributes of the garment manufacturing process and the productivity of the employees which had been collected manually and also been validated by the industry experts. Which it contains 15 attributes and 1198 fields with missing values 
	
Attribute Name	Description	Data Type
 1.Date	Date in MM-DD-YYYY	Date
 2.Day	Day of the Week	String
 3.Quarter	A portion of the month. A month was divided into four quarters	String
 4.Department	Associated department with the instance	String
 5.Team_no 	Associated team number with the instance	Numeric
 6.No_of_workers	Number of workers in each team	Float
 7.No_of_style_change	Number of changes in the style of a particular product	Numeric
 8.Targeted_productivity	Targeted productivity set by the Authority for each team for each day.	Float
 9.Smv	Standard Minute Value, it is the allocated time for a task
Work in progress. Includes the number of unfinished items for products	Float
10.Wip	Represents the amount of overtime by each team in minutes	Numeric 
11.Over_time	Represents the amount of overtime by each team in minutes	Numeric
12.Incentive	Represents the amount of financial incentive (in BDT) that enables or motivates a particular course of action.	Numeric
13.Idle_time	The amount of time when the production was interrupted due to several reasons	Float
14.Idle_men	The number of workers who were idle due to production interruption	Numeric
15.Actual_productivity	The actual % of productivity that was delivered by the workers. It ranges from 0-1.	Float

3.PySpark Installation:
Pyspark Overview: PySpark is the Python API from Apache Spark application. It is used to perform well as real-time, large-scale data processing in a distributed environment. It also provides shell in command prompt after running the proper command in prompt like figure 1 and figure 3 mention below. Spark which it is useful for analysis of data at any size in python. It supports the spark feature like data to work with DataFrame and machine learning.
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/723aecc1-ff31-4892-bf94-c9b52e835243)
In the above figure 1 is the evidence of installation of Spark-Shell from Apache application which was installed to my local desktop mentioned in the command prompt the spark version is 3.3.2 using the Scala version of 2.12.15. The Spark context Web UI available at the local host named as the PySparkShell application by clicking the link of available in the command prompt. 

Here in figure 2 is the evidence of the Sparkshell application were it used to monitor the jobs which we are running in the python programming 
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/0cb0270f-1b44-4ccd-8d0b-c24eec704987)

In Figure 3 is the evidence of the installation of the PySpark In the local desktop which gets run in the command prompt which it contains the version of the 3.3.2
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/2d9603fe-d71a-438a-a7a9-a05bcf594199)
After installation of various application of spark done in local desktop the following code for implementation is done in the Google Collaboratory (Colab) environment.

4. Implementation of PySpark to analyze the dataset and for EDA
	   PySpark set up was done to the Google Colab where it has some benefits such as pre-installed libraries and very well performing Graphics Processing Unit (GPU) for the better result of the visualization.
The code mentioned below are well formatted and used appropriate highlighter tool then code is editable
Connecting the code to the google drive:
 Code:
##Connecting to google drive and setting directory
from google.colab import drive
import os
drive.mount('/content/gdrive/')
os.chdir("/content/gdrive/")

Output:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/ce259387-0286-4bd0-bb20-b8eaaf73253e)

The above code and output snippet establishes a connection between Google Colab and Google Drive using the drive.mount() method. This enables access to files stored in Google Drive. Subsequently, the code sets the current working directory to a specific location in Google Drive using os.chdir(), facilitating seamless file operations within the Colab environment.


Installation of the findspark,pyspark and importing the libraries

Code:

!pip install -q findspark
!pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
import matplotlib.pyplot as plt
import findspark
import py4j
import pyspark
findspark.init()
print(pyspark.__version__)

OUTPUT:
 ![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/23398c40-0ae7-4ab7-8541-bc13da38213b)


This code sets up a PySpark environment for data analysis and machine learning. It installs necessary libraries, imports required modules, initializes Spark, and defines functions for classification and regression tasks. It also integrates Matplotlib for visualization. The findspark.init() command ensures proper configuration, and the final print statement displays the PySpark version of 3.4.1 and print the version of PySpark(3.4.1) and py4j(0.10.9.7)
Creating SaprkSession for the model:
Code:
data_spark = SparkSession.builder.appName("EDA_and_Models").getOrCreate()
data_spark

Output:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/3c926719-22c2-47dc-a2f9-a5c9efd8cb3c)

This code initializes a Spark session named "EDA_and_Models" for exploratory data analysis and modeling tasks. The session provides an entry point to interact with Spark's distributed computing capabilities for big data processing and analysis.






Sample code for Pyspark and Sparksession:
# Import findspark to initialize Spark environment and pyspark.sql for Spark session
import findspark
findspark.init()
from pyspark.sql import SparkSession

# Create a local Spark session and DataFrame, demonstrating data creation and display
spark = SparkSession.builder.master("local[*]").getOrCreate()
df = spark.createDataFrame([{"Hello World":"Let's create code"} for x in range(1)])

# Show the first three rows of the DataFrame without truncation
df.show(3, False)

OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/10904805-0c6a-4775-ab33-79729b234d9b)
Importing of libraries and visualize the dataset

# Import necessary libraries for data analysis and visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read a CSV file into a Spark DataFrame and convert it to a Pandas DataFrame
# for data exploration and visualization
spark_df = data_spark.read.csv("/content/sample_data/Garments", header=True, inferSchema=True)
df_gwp = spark_df.toPandas()

# Display the first five rows of the Pandas DataFrame
df_gwp.head()

OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/4b497fe5-63ec-4c3b-a371-63e59f27ba24)
This code section imports essential libraries, reads a CSV file into a Spark DataFrame, converts it to a Pandas DataFrame, and then displays the first five rows of the dataset for exploration and analysis purposes
PRE-PROCESSING AND DATA CLEANING:
Number of rows and columns
Code:
#Number of rows and columns
num_rows, num_columns = df_gwp.shape

# Print the number of rows and columns
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

Output:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/0b79adc5-d0c5-4b34-84ed-44fab8939865)
This code sets up a PySpark environment for data analysis and machine learning. It installs necessary libraries, imports required modules, initializes Spark, and defines functions for classification and regression tasks. It also integrates Matplotlib for visualization. The findspark.init() command ensures proper configuration, and the final print statement displays the PySpark version.Which it contains for 1197 rows and 15 columns.

Finding Missing values:
Code:
#Percent of missing values by columns
missing_percent = df_gwp.isnull().sum() / len(df_gwp) * 100
missing_percent
#Finding the datatype
df_gwp.dtypes

![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/f2276933-fd94-42ad-9d70-e67a047715a6)
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/7821aa3d-b633-44ea-af9c-b21009930840)
wip has 42% missing values, we handle this issue by impute the column by mean value
Handling the missing values:
Code:
# Convert selected numeric-like columns to integers, handle missing values,
# and calculate the percentage of missing values for each column
df_gwp = df_gwp.apply(pd.to_numeric, errors='ignore', downcast='integer')

# List of columns to be converted and treated for missing values
columns_to_convert = ['team', 'targeted_productivity', 'smv', 'wip', 'over_time',
                      'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'actual_productivity']

# Convert specified columns to integers and handle missing values
for col in columns_to_convert:
    df_gwp[col] = pd.to_numeric(df_gwp[col], errors='ignore', downcast='integer')

# Calculate the mean value of 'wip' column and fill missing values with the mean
mean_value = df_gwp['wip'].mean()
df_gwp['wip'].fillna(mean_value, inplace=True)

# Calculate and display the percentage of missing values in each column
missing_percent = df_gwp.isnull().sum() / len(df_gwp) * 100
missing_percent
OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/48073948-11a2-4b66-bf5e-d7d4f7e1d369)
This code block performs data preprocessing tasks: it converts selected numeric-like columns to integers, fills missing values in the 'wip' column with the mean, and calculates the percentage of missing values in each column of the DataFrame df_gwp. Now, all the attributes have no missing value. We can work on the next step - data preprocessing with descriptive analysis

Converting month and date column as string:
Code:
#Handle the date colume
# Convert the date_column to datetime if it's not already a datetime dtype
df_gwp['date'] = pd.to_datetime(df_gwp['date'])

# Extract month and year from the datetime column as strings and numbers
df_gwp['month'] = df_gwp['date'].dt.strftime('%B')
df_gwp['year'] = df_gwp['date'].dt.strftime('%Y')

#Drop 'date' colume
df_gwp = df_gwp.drop('date', axis=1, inplace=False)

# Display the first few rows of the updated DataFrame
df_gwp.head()



OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/88b08150-9059-4894-ad62-380b28b19a4a)
DATA PREPROCESSING WITH DESCRIPTIVE ANALYSIS
1.Statistical Values
CODE:
df_gwp.describe()
Output:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/fa53d062-b343-494b-bafc-6b5cc682efdc)
2.Label distributuion:
CODE:
label = df_gwp['actual_productivity']

#create figure for 2 subplots
fig, ax = plt.subplots(2,1,figsize=(9,12))

#plot the histogram
ax[0].hist(label, bins=100)
ax[0].set_ylabel('Frequency')

#add lines for the mean, median, and mode
ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)

#plot the boxplot
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('actual_productivity')

#add a title to the figure
fig.suptitle('Productivity Distribution')

fig.show()
OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/c0d830d9-0955-46f8-9d8a-507dac1bcbd0)
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/b658644a-d8f6-4a98-be5d-765638d65c0f)
This code generates a visual analysis of the distribution of the 'actual_productivity' column in the DataFrame df_gwp. It creates a two-subplot figure: the first subplot displays a histogram of the column's values with lines representing the mean and median, while the second subplot shows a boxplot to visualize the spread and outliers. The figure is titled "Productivity Distribution," providing insights into the distribution and characteristics of the 'actual_productivity' data.

3.Feature Distribution:
	In feature distribution data gets splits into two kinds of features numerical and categorical. In numerical values the attributes get into statistical function and provides some charts to visualize the data.

CODE:
numeric_features = ['team', 'targeted_productivity', 'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change']

categorical_features = ['quarter', 'department', 'day', 'month']
Distribution of the Feature in productivity attribute into two type numeric feature and categorical feature


NUMERCIAL FEATURES:
#plot the corr_matrix
corr_matrix = df_gwp.corr().round(1)

# Plot the matrix
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True, annot=True, annot_kws={"size": 12})
plt.show()


It provide the correlation matrix of heat maps distribution to all attributes. 





OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/df085d32-3cf0-4f8a-8594-73b75f2ecca1)
This heat map provides the correlation values between each attributes that inter connecting each point. The warmer colours (red and pink) indicate positive correlations, cooler colours (blue) indicate negative correlations, and lighter colours represent weaker correlations the 0 and below 0 is the weaker entity and colour which indicate the values of 0.5 to the 1 is the stronger entity where number of worker and smv has the best correlation of the values which contains the values of 0.9.

Histogram chart of every attributes to visualize the count variable:
CODE:
# Subplot grid
n_features = len(numeric_features)
n_cols = 3  # Customize the number of columns in the subplot grid
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 5))

axes = axes.flatten()

# Display the plot
for i, feature in enumerate(numeric_features):
    sns.histplot(ax=axes[i], data=df_gwp[feature], kde=True)
    axes[i].set_title(f'Histogram of {feature}')

plt.tight_layout()
plt.show()
OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/d63cb7d1-6435-4f94-9aab-c414bf96ecfd)
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/809ce59a-5625-417b-8d29-ad59e7214765)
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/91254b6d-0d76-434b-841f-c4074644984e)
This output represent the histogram chart each subplot display the distribution of numeric count for both histogram and kernel density estimate curve. The layout is designed to accommodate the given number of columns, and each subplot is labeled with the feature's name. The result is a comprehensive visualization of the distribution characteristics of each numeric feature. This code creates a grid of histograms for each numeric feature in the DataFrame df_gwp().
Categorical features:
CODE:
# Create a figure and a subplot grid
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

# Pie chart showing the percentage of each department
df_gwp['department'] = df_gwp['department'].str.lower().str.strip()
no_of_departments = df_gwp['department'].value_counts()
axes[0].pie(no_of_departments, labels=no_of_departments.index, autopct='%.1f%%')
axes[0].set_title('Percentage of Each Department')

# The relationship between each department and the label
sns.boxplot(ax=axes[1], x='department', y=label, data=df_gwp)
axes[1].set_title('Relationship Between Department and Label')
axes[1].tick_params(axis='x', rotation=45)

# Display the plot
plt.tight_layout()
plt.show()

OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/6d7b53f9-294f-4d1e-a4d9-a17cf810821f)
This code generates a figure with a subplot grid containing two visualizations related to the 'department' column and the label variable ('actual_productivity') in the DataFrame df_gwp.
The first subplot displays a pie chart showing the percentage distribution of each department. The 'department of sewing and finishing' column is preprocessed to ensure uniformity, and the pie chart visualizes the proportion of each department's presence in the dataset.The Sewing has 57.7% of contribution and 42.3% contribution in the finishing department.
The second subplot presents a boxplot that illustrates the relationship between each department and the label variable. The x-axis represents different departments, while the y-axis displays the label values. The boxplot provides insights into how the 'department' feature correlates with the label, shedding light on potential patterns or differences across different departments.
The overall figure provides a comprehensive view of the interaction between departmental distribution and the label variable.
Particular count in the department:
CODE:
no_of_departments

OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/73abf065-5ecc-49dc-a0f3-bc7c015896fe)
In sewing department we having the 691 employee and 506 employee in finishing department.
Visualization of group by day, aggregate over_time and incentive in day attribute:
CODE:
# Group by day and aggregate over_time and incentive
data_day = df_gwp.groupby('day').agg({
    'over_time': 'sum',
    'incentive': 'sum'
}).reset_index()

# Subplot grid
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

# Overtime in each day of the week
sns.barplot(ax=axes[0], x='day', y='over_time', data=data_day)
axes[0].set_title('Overtime for Each Day of the Week')

# Incentives in each day of the week
sns.barplot(ax=axes[1], x='day', y='incentive', data=data_day)
axes[1].set_title('Incentives for Each Day of the Week')

# The relationship between each day of week and label
sns.boxplot(ax=axes[2], x='day', y=label, data=df_gwp)
axes[2].set_title('Relationship Between Day of the Week and Label')

# Display the plot
plt.tight_layout()
plt.show()
OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/cc65c705-c710-4030-a3a3-1c2b5c7e9749)
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/d4b51387-96a2-4646-882c-dfaedf44286f)
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/a8e9792a-bb3f-442b-b23f-cc71e2441b3d)
The data_day DataFrame is created by grouping the data by the 'day' column and calculating the sum of 'over_time' and 'incentive' for each day.
A subplot grid with three visualizations is constructed:
The first subplot displays a bar plot illustrating the total 'over_time' for each day of the week. Where the day Thursday has over time 
The second subplot shows a bar plot representing the total 'incentive' for each day of the week. Where Monday is the day has the incentive. 
The third subplot presents a boxplot demonstrating the relationship between each day of the week and the label variable ('actual_productivity').
The combined visualizations provide insights into the trends and interactions between days of the week, overtimes, incentives, and the label variable.
Visualization of group by day, aggregate over_time and incentive in Month attribute:
CODE:
# Group by day and aggregate over_time and incentive

data_day = df_gwp.groupby('month').agg({
    'over_time': 'sum',
    'incentive': 'sum'
}).reset_index()

# Subplot grid
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

# Overtime in each day of the week
sns.barplot(ax=axes[0], x='month', y='over_time', data=data_day)
axes[0].set_title('Overtime for Each Day of the Week')

# Incentives in each day of the week
sns.barplot(ax=axes[1], x='month', y='incentive', data=data_day)
axes[1].set_title('Incentives for Each Day of the Week')

# The relationship between each day of week and label
sns.boxplot(ax=axes[2], x='month', y=label, data=df_gwp)
axes[2].set_title('Relationship Between Day of the Week and Label')

# Display the plot
plt.tight_layout()
plt.show()

OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/24f1e021-ab7e-43f6-830e-d4fbdef54a35)
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/511eb193-e861-42c3-937f-55bd0a93c172)
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/6bfeca30-69de-48ff-af7d-17028e453bad)
This code aggregates 'over_time' and 'incentive' data by month from the DataFrame df_gwp, then creates a subplot grid of three visualizations: two bar plots displaying monthly overtime and incentives, and a boxplot illustrating the relationship between months and the label variable. The overtime more was done in the January month, Incentive has more in the march month.The combined visualizations offer insights into overtime, incentives, and their impact on the label variable across different months.
Line graph for the Target productivity and actual productivity in day attribute:
CODE:
#calculate targeted productivity grouped by day
sum_targeted_productivity = df_gwp.groupby([pd.Grouper(key='day')])['targeted_productivity'].sum()

#calculate actual productivity grouped by day
sum_actual_productivity = df_gwp.groupby([pd.Grouper(key='day')])['actual_productivity'].sum()

# create the figure and axis objects
fig, ax1 = plt.subplots()

# create the first line for targeted productivity
ax1.plot(sum_targeted_productivity, color='orange')
ax1.set_xlabel('Day')
ax1.set_ylabel('Targeted Productivity', color='orange')

# create the second line for actual productivity on the same axis
ax1.plot(sum_actual_productivity, color='green')
ax1.set_ylabel('Actual Productivity', color='green')

# create a legend
ax1.legend(['Targeted Productivity', 'Actual Productivity'])

# show the plot
plt.show()
OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/84983902-2b02-4a49-8e0a-c9325ea57347)
This code computes and visualizes the daily trends of both targeted and actual productivity. It calculates the sum of targeted productivity and actual productivity grouped by day, then creates a dual-axis line plot. The orange line represents targeted productivity, and the green line represents actual productivity. The plot effectively highlights the daily variations and possible disparities between the two productivity measures. Here the actual productivity almost matches the comparison target productivity. On Sunday it slightly falls down. Meanwhile all other days are matches the targeted value. 

Line graph for the Target productivity and actual productivity by month attribute:
CODE:
#calculate targeted productivity grouped by day
sum_targeted_productivity = df_gwp.groupby([pd.Grouper(key='month')])['targeted_productivity'].sum()

#calculate actual productivity grouped by day
sum_actual_productivity = df_gwp.groupby([pd.Grouper(key='month')])['actual_productivity'].sum()

# create the figure and axis objects
fig, ax1 = plt.subplots()

# create the first line for targeted productivity
ax1.plot(sum_targeted_productivity, color='orange')
ax1.set_xlabel('month')
ax1.set_ylabel('Targeted Productivity', color='orange')

# create the second line for actual productivity on the same axis
ax1.plot(sum_actual_productivity, color='green')
ax1.set_ylabel('Actual Productivity', color='green')

# create a legend
ax1.legend(['Targeted Productivity', 'Actual Productivity'])

# show the plot
plt.show()


OUTPUT:

![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/478ad5f8-2acb-4d99-a137-c6d1eff0848a)
productivity over the month actual productivity of product is wider than the Targeted productivity which it has more profitable for the garment company. 















USING OF MACHINE LEARNING TECHNIQUES:
NORMALIZE VARIABLE:
CODE:
from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
class Standardizer(Transformer, HasInputCol, HasOutputCol):

    def __init__(self, inputCol=None, outputCol=None):
        super(Standardizer, self).__init__()
        self.setParams(inputCol = inputCol , outputCol = outputCol)

    def setParams(self, inputCol=None, outputCol=None):
      return self._set(inputCol = inputCol, outputCol = outputCol)

    def _transform(self, dataset):
      from pyspark.sql.functions import stddev, mean, col
      out_col = self.getOutputCol()
      in_col = dataset[self.getInputCol()]
      xmin, xmax = dataset.select(min(in_col), max(in_col)).first()
      return dataset.withColumn(out_col, (in_col - xmin)/(xmax-xmin))
# iterates the num_input list, transformer for each column and appending to standardizers list
#standardizers
standardizers = [Standardizer(inputCol = column, outputCol = column+"_standardized") for column in num_input] 

Encode Dummy Variables
Encoding the fake values for the indexer values 
indexers = [StringIndexer(inputCol = column, outputCol = column+"_index") for column in cat_input]
encoders = [OneHotEncoder(inputCol = column+"_index", outputCol = column+"_dummy") for column in cat_input]


Combine Stages
#from pyspark.ml.classification import *
import functools
import operator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

#input_cols
input_cols = []
for i in cat_input:
  input_cols.append(i+"_dummy")
for i in num_input:
  input_cols.append(i+"_standardized")
#stages
stages = []
assembler = VectorAssembler(inputCols= input_cols, outputCol="features") #concatenate all input variables and names as features [[0,1,0],30,20,40000]
stages = functools.reduce(operator.concat, [indexers, encoders, standardizers]) #indexers,  encoders, standardizers])
stages.append(assembler)


Create Spark Pipeline
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=stages)
spark_clean_df = pipeline.fit(spark_clean_df).transform(spark_clean_df)


SPLIT TRAIN AND TEST SET
# 70% to the training and 30% to the testing set with random seed reproducibility.
train, test = spark_clean_df.randomSplit([0.7, 0.3], seed = 2008) 



MODEL FITTING
from pyspark.ml.regression import *
!pip install findspark

1. Linear Regression

CODE:
# create logistic regression
lr = LinearRegression(labelCol=target, featuresCol="features", maxIter=10) 
# Linear regressiion model with 10 maximum number of iteration for optimization algorithm.
lr_model = lr.fit(train)

#tranform model to test
lr_result = lr_model.transform(test)
lr_result.select('actual_productivity', 'prediction','features').show(5)
OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/15b0bfcd-79e7-46dc-af12-500170ce3f19)

Here is the highest accuracy score for the prediction in actual productivity is 78% for the Linear Regression.

2. Decision Tree:

CODE:
# create decision tree
dt = DecisionTreeRegressor(labelCol=target, featuresCol="features")
dt_model = dt.fit(train)

dt_result = dt_model.transform(test)
dt_result.select('actual_productivity', 'prediction', 'features').show(5)



OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/c9ee38f2-02be-4867-b2a4-c5e41eafae24)

Here is the highest accuracy score for the prediction in actual productivity is 83% for the Decision tree.

3. Random Forest

CODE:
# create random forest
rf = RandomForestRegressor(labelCol=target, featuresCol="features", numTrees=10) #random forest regression with 10 decision tree
rf_model = rf.fit(train)

rf_result = rf_model.transform(test)
rf_result.select('actual_productivity', 'prediction', 'features').show(5)

OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/84b97450-c117-4dae-9dca-ae25cb9834b1)

Here is the highest accuracy score for the prediction in actual productivity is 83% for the Random Forest.


4. Gradient Boosting:

CODE:
# create gradient boosting model to train set
gbt = GBTRegressor(labelCol=target, featuresCol="features", maxIter=10) #gradient boosted tree regressor with 10 maximum number of iteration for optimization algorithm
gbt_model = gbt.fit(train)

gbt_result = gbt_model.transform(test) #tranform model
gbt_result.select('actual_productivity', 'prediction','features').show(5)

OUTPUT:

 

Here is the highest accuracy score for the prediction in actual productivity is 84% for the Gradient Boosting.

Comparing to all other method of machine learning “Gradient Boosting” suits the best model to find the accuracy level for the productivity. 








PERFORMANCE EVALUATION
Model Evaluation Metrics
CODE:
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator

models = [lr_model, dt_model, rf_model, gbt_model]
model_names = ['Linear Regression','Decision Tree' ,'Random Forest', 'Gradient Boosting']

# define the metrics
metric_names = ['R2', 'RMSE', 'MSE', 'MAE']
metric_values = {name: [] for name in metric_names}

# calculate metrics for each model
for model, name in zip(models, model_names):
    # calculate R2
    r2_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=target, metricName="r2")
    r2 = r2_evaluator.evaluate(model.transform(test))
    metric_values['R2'].append(r2)

    # calculate RMSE
    rmse_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=target, metricName="rmse")
    rmse = rmse_evaluator.evaluate(model.transform(test))
    metric_values['RMSE'].append(rmse)

    # calculate MSE
    mse_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=target, metricName="mse")
    mse = mse_evaluator.evaluate(model.transform(test))
    metric_values['MSE'].append(mse)

    # calculate MAE
    mae_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=target, metricName="mae")
    mae = mae_evaluator.evaluate(model.transform(test))
    metric_values['MAE'].append(mae)

df_metrics = pd.DataFrame(metric_values, index=model_names)
print(df_metrics)



OUTPUT:
 
The lowest RMSE, MSE, and MAE values, indicating the best overall performance models. R2 is response variable that exaplained by independent variables. In other words, it represents how well the model fits the data. In this case, Gradient Boosting is the best-performing model among evaluated models.

Model Evaluation Plots
CODE:
actual_values = lr_result.select('actual_productivity').toPandas()

lr_predicted_values = lr_result.select('prediction').toPandas()
dt_predicted_values = dt_result.select('prediction').toPandas()
rf_predicted_values = rf_result.select('prediction').toPandas()
gbt_predicted_values = gbt_result.select('prediction').toPandas()

import numpy as np

# Create an array of index values for the x-axis
index_values = np.arange(len(actual_values))

# Create subplots
fig, axes = plt.subplots(1, 4, figsize=(25, 5), sharey=True)

# Linear Regression plot
axes[0].scatter(index_values, actual_values, color='blue', label='Actual Values')
axes[0].scatter(index_values, lr_predicted_values, color='red', label='Predicted Values', alpha=0.5)
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Values')
axes[0].set_title('Linear Regression')
axes[0].legend()

# Decision Tree plot
axes[1].scatter(index_values, actual_values, color='blue', label='Actual Values')
axes[1].scatter(index_values, dt_predicted_values, color='red', label='Predicted Values', alpha=0.5)
axes[1].set_xlabel('Index')
axes[1].set_title('Decision Tree')
axes[1].legend()

# Random Forest plot
axes[2].scatter(index_values, actual_values, color='blue', label='Actual Values')
axes[2].scatter(index_values, rf_predicted_values, color='red', label='Predicted Values', alpha=0.5)
axes[2].set_xlabel('Index')
axes[2].set_title('Random Forest')
axes[2].legend()

# Gradient Boosting plot
axes[3].scatter(index_values, actual_values, color='blue', label='Actual Values')
axes[3].scatter(index_values, gbt_predicted_values, color='red', label='Predicted Values', alpha=0.5)
axes[3].set_xlabel('Index')
axes[3].set_title('Gradient Boosting')
axes[3].legend()

plt.show()
OUTPUT:
4. Gradient Boosting:

CODE:
# create gradient boosting model to train set
gbt = GBTRegressor(labelCol=target, featuresCol="features", maxIter=10) #gradient boosted tree regressor with 10 maximum number of iteration for optimization algorithm
gbt_model = gbt.fit(train)

gbt_result = gbt_model.transform(test) #tranform model
gbt_result.select('actual_productivity', 'prediction','features').show(5)

OUTPUT:

 

Here is the highest accuracy score for the prediction in actual productivity is 84% for the Gradient Boosting.

Comparing to all other method of machine learning “Gradient Boosting” suits the best model to find the accuracy level for the productivity. 








PERFORMANCE EVALUATION
Model Evaluation Metrics
CODE:
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator

models = [lr_model, dt_model, rf_model, gbt_model]
model_names = ['Linear Regression','Decision Tree' ,'Random Forest', 'Gradient Boosting']

# define the metrics
metric_names = ['R2', 'RMSE', 'MSE', 'MAE']
metric_values = {name: [] for name in metric_names}

# calculate metrics for each model
for model, name in zip(models, model_names):
    # calculate R2
    r2_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=target, metricName="r2")
    r2 = r2_evaluator.evaluate(model.transform(test))
    metric_values['R2'].append(r2)

    # calculate RMSE
    rmse_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=target, metricName="rmse")
    rmse = rmse_evaluator.evaluate(model.transform(test))
    metric_values['RMSE'].append(rmse)

    # calculate MSE
    mse_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=target, metricName="mse")
    mse = mse_evaluator.evaluate(model.transform(test))
    metric_values['MSE'].append(mse)

    # calculate MAE
    mae_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=target, metricName="mae")
    mae = mae_evaluator.evaluate(model.transform(test))
    metric_values['MAE'].append(mae)

df_metrics = pd.DataFrame(metric_values, index=model_names)
print(df_metrics)



OUTPUT:
 
The lowest RMSE, MSE, and MAE values, indicating the best overall performance models. R2 is response variable that exaplained by independent variables. In other words, it represents how well the model fits the data. In this case, Gradient Boosting is the best-performing model among evaluated models.

Model Evaluation Plots
CODE:
actual_values = lr_result.select('actual_productivity').toPandas()

lr_predicted_values = lr_result.select('prediction').toPandas()
dt_predicted_values = dt_result.select('prediction').toPandas()
rf_predicted_values = rf_result.select('prediction').toPandas()
gbt_predicted_values = gbt_result.select('prediction').toPandas()

import numpy as np

# Create an array of index values for the x-axis
index_values = np.arange(len(actual_values))

# Create subplots
fig, axes = plt.subplots(1, 4, figsize=(25, 5), sharey=True)

# Linear Regression plot
axes[0].scatter(index_values, actual_values, color='blue', label='Actual Values')
axes[0].scatter(index_values, lr_predicted_values, color='red', label='Predicted Values', alpha=0.5)
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Values')
axes[0].set_title('Linear Regression')
axes[0].legend()

# Decision Tree plot
axes[1].scatter(index_values, actual_values, color='blue', label='Actual Values')
axes[1].scatter(index_values, dt_predicted_values, color='red', label='Predicted Values', alpha=0.5)
axes[1].set_xlabel('Index')
axes[1].set_title('Decision Tree')
axes[1].legend()

# Random Forest plot
axes[2].scatter(index_values, actual_values, color='blue', label='Actual Values')
axes[2].scatter(index_values, rf_predicted_values, color='red', label='Predicted Values', alpha=0.5)
axes[2].set_xlabel('Index')
axes[2].set_title('Random Forest')
axes[2].legend()

# Gradient Boosting plot
axes[3].scatter(index_values, actual_values, color='blue', label='Actual Values')
axes[3].scatter(index_values, gbt_predicted_values, color='red', label='Predicted Values', alpha=0.5)
axes[3].set_xlabel('Index')
axes[3].set_title('Gradient Boosting')
axes[3].legend()

plt.show()
OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/8c32af25-ba1e-4dc5-b270-63db1634d67e)
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/8f730a5a-bb5c-4af0-9ea7-6e7df84d7bce)
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/27174fce-5c5d-4375-8fcb-9d20b4f7e688)
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/2b70a940-ad23-4553-8478-2ffa537883a9)
Model Residual Plots
# Calculate residuals
lr_residuals_df = lr_result.withColumn('residuals', lr_result.actual_productivity - lr_result.prediction)
dt_residuals_df = dt_result.withColumn('residuals', dt_result.actual_productivity - dt_result.prediction)
rf_residuals_df = rf_result.withColumn('residuals', rf_result.actual_productivity - rf_result.prediction)
gbt_residuals_df = gbt_result.withColumn('residuals', gbt_result.actual_productivity - gbt_result.prediction)

import seaborn as sns

# Convert residuals to Pandas
lr_residuals = lr_residuals_df.select('residuals').toPandas()
dt_residuals = dt_residuals_df.select('residuals').toPandas()
rf_residuals = rf_residuals_df.select('residuals').toPandas()
gbt_residuals = gbt_residuals_df.select('residuals').toPandas()

# Create combined DataFrame for plotting
residuals_df = pd.concat([
    lr_residuals.assign(model='Linear Regression'),
    dt_residuals.assign(model='Decision Tree'),
    rf_residuals.assign(model='Random Forest'),
    gbt_residuals.assign(model='Gradient Boosting')
], ignore_index=True)

# Create violin plot
plt.figure(figsize=(13, 6))
sns.violinplot(data=residuals_df, x='model', y='residuals', inner='quartile')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals Distribution for Models')
plt.show()

OUTPUT:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/a74d1820-e9e0-4577-80d0-6a8e1072f2a9)
Residual distribution is approximately symmetrical around zero, which indicates that the model predictions are likely to be above or below the actual productivity values. Quartile lines inside the violin indicate the range of the residuals, and the red dashed line represents the zero line, which indicates that model predicted the actual productivity value perfectly.

Residuals
CODE:
from pyspark.sql.functions import mean, stddev, min, max

# Calculate residuals summary statistics for models
lr_summary_df = lr_residuals_df.select(mean('residuals').alias('mean'), stddev('residuals').alias('stddev'), min('residuals').alias('min'), max('residuals').alias('max'))
dt_summary_df = dt_residuals_df.select(mean('residuals').alias('mean'), stddev('residuals').alias('stddev'), min('residuals').alias('min'), max('residuals').alias('max'))
rf_summary_df = rf_residuals_df.select(mean('residuals').alias('mean'), stddev('residuals').alias('stddev'), min('residuals').alias('min'), max('residuals').alias('max'))
gbt_summary_df = gbt_residuals_df.select(mean('residuals').alias('mean'), stddev('residuals').alias('stddev'), min('residuals').alias('min'), max('residuals').alias('max'))

# print residuals summary statistics for models
print("Linear Regression Residuals:")
lr_summary_df.show()
print("Decision Tree Residuals:")
dt_summary_df.show()
print("Random Forest Residuals:")
rf_summary_df.show()
print("Gradient Boosting Residuals:")
gbt_summary_df.show()

Output:
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/6c0f551d-f526-4a72-9112-1b1b75709dfd)
Overall models, the mean of the residuals is very close to zero and standard deviation is small, which indicates that the models are performing well in terms of predicting (actual productivity) values.



                                                    Data Visualization using tableau 

Evidence of Installation tableau:
Installation done in local desktop and directory mention clearly.
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/13203eb8-5bb5-45cf-b69c-693cfcfaf9eb)
Preprocessing Data 
1.Cleaning the data by using data interpreter by clicking the check box 
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/cc5c3bcc-ed42-41e9-bd4c-9f464fc2360b)
2.Replacing the null values with the zero:
In the dataset it contains null values in the column of wip(work in progress) by using ZN function in the calculated field replacement done to null values with zero.

![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/dd3d9d43-d02a-4852-9bda-df202ef79d45)
Figure 3 calculated field using zn function
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/1285c4db-5a17-4f70-b7b3-f9b9024c7dfe)
Figure 4 show the replacement of null with zero

BIVARIATE ANALYSIS ON TABLEAU
1. Actual and Target productivity following by month of date: This area charts shows the analysis of productivity falls down from initial month January to the march.
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/48b531e0-4e60-4e94-8cf7-3c72f089ad7d)
Figure 5 shows productivity followed by month of date by using area chart

2. Actual and Target productivity following by day: Here we can see the comaprison of actual and targeted productivity by using horizontal bar in both we can see the Wednesday were more productivity work goes on. 
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/39377127-df70-47aa-92de-b8ca81a4edfa)
Figure 6 shows productivity by day using horizontal bar

3.Productivity by the department classified by day: Here we mentioned two department finishing process and sewing process which it was classified by the days. Here sewing process plays the major process by following Wednesday at its peak by using scatter plot.
![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/20908da5-d6f6-4ca5-8d93-008e4ad53ab3)
4.Department working through over time: Here we can see the department of sewing works for the over time to increase the productivity.

![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/845aad89-102a-40b8-8d3f-3e1f3ec627c5)
5.Incentive by monthly: This side by side chart show the level of incentive gets by the worker for each month.

![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/4e455a9e-da05-4a9b-95c6-764537c66b82)
Figure 9 show the side by side charts for incentive.

6.Number worker in departments: This highlighted table shows the number of workers who works in their own department in sewing 691 and in finishing 506

![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/8309feab-e089-495b-8929-7ab80b82a807)
7.Work in progress measured by the days: Here the below bullet graph shows the average and count for work in progress followed by the days. Were Monday have major part working in progress. 

![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/f619dc7f-614d-4800-a8a4-1cb92eba4bf8)


The Main Dashboard for the data visualization
This the over view of all the charts that we create for the visualization.


![image](https://github.com/Yogesh-653/Big-Data-Analytics-and-Data-Visualization/assets/60870157/b89d8f2c-a1ce-4ce3-bc8c-3a2107d6ddd0)


DISCUSSION: 
Garment productivity plays a pivotal role in the textile and apparel industry, influencing both operational efficiency and overall profitability. It refers to the rate at which garments are produced within a specific timeframe while maintaining quality standards. The complex nature of the industry, involving various stages from design to manufacturing, makes understanding and optimizing garment productivity essential for success.
The initial productivity of the garment the data showed has mixed of all attributes and values are highly variability.
In this project we have 4 machine learning model were tested both classification and regression they are linear regression, Random forest, Gradient Boosting Trees, Decision Tree. This model evaluate the Residuals, R-Squared(R2), Mean Squared Error(MSE), Root mean squared error(RMSE),Mean Absolute Error(MAE). These model are used to indicate the best overall performance.
The best performing model in all case is Gradient Boosting Tree which overcome the lowest error values and highest R-squared and means accuracy level. According to the overall performance models.

CONCLUSION:
In this project done with working model of the PySpark library by loading the dataset and pre-processing the data. Then covering all EDA process for the following dataset using appropriate classification and regression algorithm for the best model fitting. Then Creating dashboard for the visualization the dataset and finding the best model in the model prediction done in this project.


