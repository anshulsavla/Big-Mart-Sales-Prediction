#!/usr/bin/env python
# coding: utf-8

# # Sales Prediction for Big Mart Outlets
# 
# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and predict the sales of each product at a particular outlet.
# 
# Using this model, BigMart will try to understand the properties of products and outlets which play a key role in increasing sales.
# 
# Please note that the data may have missing values as some stores might not report all the data due to technical glitches. Hence, it will be required to treat them accordingly. 

#  # Let's Start !

# In[1]:


#Importing the basic libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
plt.style.use('seaborn-whitegrid')
sns.set_style("white")


# For integrity purpose , we wont take a look at the testing dataset . We will be dealing with the testing dataset , once we are absolutely done, with our training dataset

# In[2]:


#Reading the training and testing dataset

df = pd.read_csv('train_bigmart.csv')
df_test = pd.read_csv('test_bigmart.csv')
df.head()


# In[3]:


#To check the descriptive statistics of our dataset
df.describe()


# In[4]:


#Now we will extract the categorical features from our dataset
cat_features = [index for index in df.columns if len(df[index].unique())<25]
cat_features


# To know how many unique values are there in each categorical feature i will define a function named counts that will return the value counts of differnt unique values of each categorical feature 

# In[5]:


def counts (feature):
    return df[feature].value_counts()
counts('Outlet_Type')


# we can see that  that there is mistake in collection the data for  Item_Fat_Content .
# 
# For Low Fat they have used three symbols which are Low Fat, LF an low fat so we will replace that values by original values that areLow fat and regular

# In[6]:


df['Item_Fat_Content'].replace('LF','Low Fat',inplace=True)
df['Item_Fat_Content'].replace('low fat','Low Fat',inplace=True)
df['Item_Fat_Content'].replace('reg','Regular',inplace=True)


# In[7]:


#For better understanding and visualisation purpose i will now create the countplot for all the categorical features
for i in cat_features:
    plt.figure(figsize=(10,6))
    df[i].value_counts().plot(kind='bar')
    plt.xlabel(i)
    plt.ylabel('Value_Counts')
    plt.show()


# # Filling the missing values

# In[8]:


#As there is no zero visibility of any item in any kind of shopping centre

df['Item_Visibility'].replace(0,np.nan,inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df['Outlet_Size'].fillna('Unknown',inplace=True)


# In[11]:


plt.figure(figsize=(8,6))
sns.barplot(df['Outlet_Size'],df['Item_Outlet_Sales'])
plt.show()


# In[12]:


plt.figure(figsize=(8,6))
sns.barplot(df['Outlet_Size'],df['Item_Visibility'])
plt.show()


# From above two graph we can conclude as follows:
# 
# Graph 1 : Generally there is low outlet sales in stores having outlet size small and unknown . Also store having small outlet size resembles the store having unknown outlet size
# 
# Graph 1 : Generally there is high visibility in stores having outlet size small and unknown . Also store having small outlet size resembles the store having unknown outlet size

# In[13]:


a=pd.crosstab(df['Outlet_Size'],df['Outlet_Type'])
a.plot(kind='bar')
plt.show()


# Above graph also tells us that outlet having small size resembles the outlet having unknown size.Also outlet having smaller size are supermaket type 1 or grocery store

# In[14]:


df1=df[df['Outlet_Size'] == 'Unknown']
df1.head()


# This dataframe also supports the previous reason that outlet having unknown size are of type grocery store or supermarket type 1

# So i'll fill the missing values of outlet size by small as there are enough evidence

# In[15]:


df['Outlet_Size'].replace('Unknown','Small',inplace=True)


# In[16]:


df0=df[df['Item_Identifier'] == 'FDU28']
df0


# If you look the above dataframe for particular identifier,values for item visibility are same for outlet of type supermarket and for grocery store the value is old value + 0.1

# In[17]:


df['Outlet_Type1'] = df['Outlet_Type']


# In[18]:


#converting supermarket type 1 ,2,3 to supermarket
dict={'Supermarket Type1':'Supermarket','Supermarket Type2':'Supermarket','Supermarket Type3':'Supermarket','Grocery Store':'Grocery Store'}
df['Outlet_Type1'] = df['Outlet_Type1'].map(dict)
df.isnull().sum()


# In[19]:


#Filling the missing value wrt item identifier and outlet type
df['Item_Visibility'].fillna(df.groupby(['Item_Identifier','Outlet_Type1'])['Item_Visibility'].transform('mean'),inplace=True)


# In[20]:


df.isnull().sum()


# After filling the missing value wrt item identifier and outlet type there are several values which are still nan because if for particular identifier if there is only one type of outlet type which is nan we cannot take mean of it and fill that value

# In[21]:


df7=df[df['Item_Visibility'].isnull()]
df7.head()


# All missing value corresponds to outlet of type grocery store.
# 
# As we have observed that item visibility for grocery store is plus 0.1 the value for supermarket for particular item identifier

# In[22]:


df['Item_Visibility'] = (df.groupby('Item_Identifier')['Item_Visibility'].transform(lambda x:x.fillna(x.mean()+0.1)))


# After filling the missing values for item identifier.
# 
# Now I'll try to fill the missing values present in Item_Weight column
# 
# To fill the missing values i'll the dependance of item weight with various categorical features

# In[23]:


plt.figure(figsize=(8,6))
sns.barplot(df["Item_Fat_Content"],df['Item_Weight'])
plt.show()


# Generaaly Low Fat product have higher item weight compared to regular product.
# 
# But if you look the at above graph the difference is very small to generalize our point

# In[24]:


df2=df[df['Item_Identifier'] == 'FDA15']
df2.head()


# In[25]:


df3=df[df['Item_Identifier'] == 'FDA08']
df3.head()


# If you look at the above two dataframe you can clearly say that item identifier is strongly correlated with item weight so i will fill the missing value of item weight wrt item identifier

# In[26]:


df['Item_Weight'].fillna(df.groupby('Item_Identifier')['Item_Weight'].transform('median'),inplace=True)


# In[27]:


df.isnull().sum()


# In[28]:


df4=df[df['Item_Weight'].isnull()]
df4


# After filling missing values wrt item identifier there are 4 unique identifiers present so we need to still fill this remaining 4 values.
# 
# So i will fill this missing values wrt item type

# In[29]:


df['Item_Weight'].fillna(df.groupby('Item_Type')['Item_Weight'].transform('mean'),inplace=True)


# # Removing the skewness

# In[30]:


num_features = ['Item_Weight','Item_Visibility','Item_MRP']
for i in num_features:
    sns.distplot(df[i],bins=8)
    plt.xlabel(i)
    plt.show()


# From above graphs, graph 2 is positively skewed so i will remove the skewness by performing log transformation

# In[31]:



df['log_visibility'] = np.log(df['Item_Visibility'])


# # Boxplots

# In[32]:


for i in num_features:
    sns.boxplot(df[i])
    plt.xlabel(i)
    plt.show()


# In[33]:


def plot_graph(x_axis_feature,type_of_graph,hue=None):
    for i in num_features:
        if (type_of_graph=='boxplot'):
            sns.boxplot(x_axis_feature,i,hue=hue,data=df)
            plt.show()
        elif (type_of_graph=='barplot'):
             sns.barplot(x_axis_feature,i,data=df)
             plt.show()
        elif (type_of_graph=='catplot'):
             sns.barplot(x_axis_feature,i,hue=hue,data=df)
             plt.show()
plot_graph('Outlet_Size','boxplot')


# As there are outliers present in item visibility and item outlet sales i will try to remove the outliers with the help of z score.

# # Removing the Outliers

# In[34]:


# Method 1 - Removing outliers with the help of z score

def remove_outlier(column):
    std = np.std(df[column])
    mean = np.mean(df[column])
    outlier = []
    for i in df[column]:
        zscore = (i - mean) / std
        
        #Considering z>3 because z>3 sinifies 99.7%values fall in that region
        
        if(zscore > 3):
            outlier.append(i)
            minimum = np.min(outlier)
    return minimum


# In[35]:


print(remove_outlier('Item_Visibility'))


# So all the values above 0.22 will be consiered as an outlier using z score

# In[36]:


# Method - 2 Removing the outlier with the help of IQR score

q3 = df['Item_Visibility'].quantile(0.85)
q1 = df['Item_Visibility'].quantile(0.15)

#IQR Score


IQR = q3 - q1

print(q3 + 1.5 * IQR)


# Interquartile Range (IQR) is important because it is used to define the outliers. It is the difference between the third quartile and the first quartile (IQR = Q3 -Q1). Outliers in this case are defined as the observations that are below (Q1 − 1.5x IQR) or boxplot lower whisker or above (Q3 + 1.5x IQR) or boxplot upper whisker.

# So all the values above 0.20 will be considered as an outlier using IQR score

# In[37]:


# Method 3 - Using Standard Deviation
m = (df['Item_Visibility'].mean())
s = (df['Item_Visibility'].std())
print(m+(3*s))


# If data distribution is approx normal then about 68% values lie within one standard deviation of mean and about 95% lie within two standard deviation and about 99.7% values lie within three standard deviation.
# 
# Therefore, if you have any data point that is more than 3 times standard deviation then those points are likely to be considered as outlier.
# 
# So all the values above 0.22217 will be considered as an outlier using SD method

# In[38]:


a = df[(df['Outlet_Type'] == 'Grocery Store')]
c = a['Item_Visibility'].median()
c


# Actually all the outliers present in item visibility column corresponds to outltet of type grocery store so we will replace the outliers values by the median of item visibility which corresponds to outlet type grocery store

# ### Now we will replace those values which are considered as an outliers using zscore

# In[39]:


df['Item_Visibility']=df['Item_Visibility'].where(df['Item_Visibility']<=0.2799065748499999,0.101231721)


# # Feature Engineering

# In[40]:


df.describe()


# In[41]:


sns.lmplot('Item_MRP','Item_Outlet_Sales',data=df)
plt.show()


# In[42]:


# I will bin Item_MRP and Item_Weight into 3 categories using descriptive statistics
df['MRP_bin']=pd.cut(df['Item_MRP'],bins=[31,93.8265,143.0128,185.6437,267],labels=['cheap','affordable','slightly expensive','expensive'])
df['Weight_bin']=pd.cut(df['Item_Weight'],bins=[4,8.785,12.65,16.85,22],labels=['vlight','light','moderate','heavy'])


# I will not bin the item visibility column as it is positively skewed

# In[43]:


#Extracting first two letters of Item_Identifier
df['Item_Identifier_temp']=df['Item_Identifier'].str[:2]
df.head()


# FD - Foods
# 
# DR - Drinks
# 
# NC - Non Consumable

# In[44]:


sns.barplot(df['Outlet_Type'],df['Item_Outlet_Sales'])
plt.show()


# In[45]:


a=pd.crosstab(df['Outlet_Type'],df['Outlet_Establishment_Year'])
a.plot(kind='bar')
plt.show()


# Since  more no of supermarket type 3 are built in 1987 than grocery store is also one of the reason why supermarket type 3 has higher sales.
# 
# Another reason is why would people go to grocery store if they get all their desired product at one stop and even at a cheaper rate

# In[46]:


sns.barplot(df['Outlet_Type'],df['Item_Visibility'])
plt.show()


# Visibility of all products are same in supermarket of type 1 , 2 , 3 since most of the supermarket have same intetior or the other reason might be that all the supermarket are same and are just the different branches

# In[47]:


merge=['Outlet_Location_Type','Outlet_Size','Item_Identifier_temp']
for i in merge:
    a=pd.crosstab(df['Outlet_Type'],df[i])
    a.plot(kind='bar')
    plt.show()


# All of the graph proves that the behaviour of supermarket type 2 and 3 are similar so we will add those types

# In[48]:


dict={'Supermarket Type1':'Supermarket Type1','Supermarket Type2':'Supermarket Type2','Supermarket Type3':'Supermarket Type2','Grocery Store':'Grocery Store'}
df['Outlet_Type']=df['Outlet_Type'].map(dict)


# In[49]:


df['Outlet_Existence']=[2020-i for i in df['Outlet_Establishment_Year']]
df.head()


# Outlet Existence shows how old that outlet is.

# In[50]:


df['Outlet_Status']=[0 if i<=2000 else 1 for i in df['Outlet_Establishment_Year']]
df.head()


# Classifying outlet into 2 categories 0(old) and 1(new) according to the year in which the store was built

# In[51]:


sns.barplot(df['Outlet_Status'],df['Item_Outlet_Sales'])
plt.show()


# Above graph shows that newer outlet have higher outlet sales

# In[52]:


a=pd.crosstab(df['Outlet_Status'],df['Outlet_Type'])
a.plot(kind='bar')
plt.show()


# Reason for the higher sales of newly built outlets are that newly built outlet are only of type supermarket and none of are grocery store

# In[53]:


sns.barplot(df['Outlet_Status'],df['Item_Visibility'])
plt.show()


# In[54]:


a=pd.crosstab(df['Outlet_Status'],df['Outlet_Size'])
a.plot(kind='bar')
plt.show()


# Generally newer outlet have low item visibility since most of the newer outlets are small in size

# In[55]:


# Since Non Consumables cannot be low fat or regular so  will create new var in item fat content for non consumable item identifier
df.loc[df['Item_Identifier_temp']=='NC','Item_Fat_Content']='No Fat'
df['Item_Fat_Content'].value_counts()


# In[56]:


for i in df['Item_Type'].unique():
    for j in df['Item_Type'].unique():
        if(i!=j):
            print(i,j,df.loc[((df['Item_Type']==i) | (df['Item_Type']==j))]['Item_Outlet_Sales'].mean())


# I've written this lines of code to show that which two items make the best combination in terms of higer outlet sales.
# 
# This code can be later modified to make combination of n items

# In[57]:


#To check which item type are classified under DR , NC , FD
a=df.groupby('Item_Identifier_temp')['Item_Type'].value_counts()
a.head(60)


# In[58]:


df['Item_Type'].unique()


# In[59]:


# Craeted a new column named calorie count which will show the calorie of corresponding item type
def set_cal(df):
    if df['Item_Type']=='Dairy':
        return 46
    elif df['Item_Type']=='Soft Drinks':
        return 51
    elif df['Item_Type']=='Meat':
        return 143
    elif df['Item_Type']=='Fruits and Vegetables':
        return 65
    elif df['Item_Type']=='Baking Goods':
        return 140
    elif df['Item_Type']=='Snack Foods':
        return 475
    elif df['Item_Type']=='Frozen Foods':
        return 50
    elif df['Item_Type']=='Breakfast':
        return 350
    elif df['Item_Type']=='Hard Drinks':
        return 250
    elif df['Item_Type']=='Canned':
        return 80
    elif df['Item_Type']=='Starchy Foods':
        return 90
    elif df['Item_Type']=='Seafood':
        return 204
    elif df['Item_Type']=='Breads':
        return 250
    else:
        return 0
df['Calorie_Count_per_100g']=df.apply(set_cal,axis=1)


# In[60]:


df['Calorie_Count_per_givenwt']=df['Calorie_Count_per_100g']/100


# In[61]:


df['Calorie_Count_per_givenwt']=df['Calorie_Count_per_givenwt']*df['Item_Weight']
df.head()


# In[62]:


# I will plot the histogram of my new feature that is calorie count
df['Calorie_Count_per_givenwt'].plot(kind='hist',bins=8)


# In[63]:


#As the above graph is positively skewed i will remove the skewness ny doing log transformation
df['log_Calorie']=np.sqrt(df['Calorie_Count_per_givenwt'])


# In[64]:


# To check the value counts of all item type to know which items are bought daily and which are boght rarely 
b=df.groupby('Outlet_Identifier')['Item_Type'].value_counts()
b.head(30)


# In[65]:


# Grouping item type into D(daily) and S(sometimes) according to daily need
dict={'Dairy':'D','Meat':'S','Fruits and Vegetables':'D','Breakfast':'S','Breads':'S','Starchy Foods':'S','Seafood':'S','Soft Drinks':'S','Household':'D','Baking Goods':'S','Snack Foods':'D','Frozen Foods':'D','Hard Drinks':'S','Canned':'D','Health and Hygiene':'S','Others':'S'}
df['Item_Frequency']=df['Item_Type'].map(dict)


# In[66]:


df.head()


# In[67]:


sns.barplot(df['MRP_bin'],df['Item_Outlet_Sales'])
plt.show()


# In[68]:


sns.barplot(df['Item_Frequency'],df['Item_Outlet_Sales'])
plt.show()


# The above graph is quite clear since the graph shows that daily used products generally have higher outlet sales

# In[69]:


a=pd.crosstab(df['Item_Frequency'],df['Outlet_Identifier'])
a.plot(kind='bar')
plt.show()


# In[70]:


a=pd.crosstab(df['Outlet_Type'],df['Outlet_Identifier'])
a.plot(kind='bar')
plt.show()


# First graph shows that OUT019 and OUT010 are two outles which have least sales of daily used products.
# 
# Second graph shows that both those outlets corresponds to outlet of type grocery store.
# 
# By combining the result of both graphs we can justify that why grocery store has least outlet Sales
# 
# So from above graphs we found another reason why grocery store has minimum sales. 

# # Label Encoding

# In[71]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Item_Identifier_temp']=le.fit_transform(df['Item_Identifier_temp'])
df['Item_Type']=le.fit_transform(df['Item_Type'])
df['Outlet_Type']=le.fit_transform(df['Outlet_Type'])
df['Outlet_Identifier']=le.fit_transform(df['Outlet_Identifier'])
df['Outlet_Type1']=le.fit_transform(df['Outlet_Type1'])
dict={'Small':0,'Medium':1,'High':2}
df['Outlet_Size']=df['Outlet_Size'].map(dict)
dict1={'cheap':0,'affordable':1,'slightly expensive':2,'expensive':3}
df['MRP_bin']=df['MRP_bin'].map(dict1)
dict2={'vlight':0,'light':1,'moderate':2,'heavy':3}
df['Weight_bin']=df['Weight_bin'].map(dict2)
dict3={'Tier 3':0,'Tier 2':1,'Tier 1':2}
df['Outlet_Location_Type']=df['Outlet_Location_Type'].map(dict3)
dict4={'No Fat':0,'Low Fat':1,'Regular':2}
df['Item_Fat_Content']=df['Item_Fat_Content'].map(dict4)
dict5={35:0,33:1,23:2,22:3,21:4,18:5,16:6,13:7,11:8}
df['Outlet_Existence']=df['Outlet_Existence'].map(dict5)
dict6={'D':1,'S':0}
df['Item_Frequency']=df['Item_Frequency'].map(dict6)
dict7={1985:0,1987:1,1997:2,1998:3,1999:4,2002:5,2004:6,2007:7,2009:8}
df['Outlet_Establishment_Year']=df['Outlet_Establishment_Year'].map(dict7)


# In[72]:


# Final Dataset ready for training the model
pd.set_option('display.max_columns',None)
df.head()


# In[73]:


df.describe()


# In[74]:


df.columns


# # Feature Selection

# In[75]:


y=df[['Item_Outlet_Sales']]
X=df[[ 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type', 'log_visibility',
       'MRP_bin', 'Weight_bin', 'Item_Identifier_temp', 'Outlet_Existence',
       'Outlet_Status', 'Calorie_Count_per_100g', 'Calorie_Count_per_givenwt',
       'log_Calorie', 'Item_Frequency']]
x=df[['MRP_bin','Weight_bin','Item_Fat_Content','Item_Type', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type','Item_Identifier_temp', 'Outlet_Existence',
       'Outlet_Status']]


# # Train Test Split

# Using train test split to create training set and validation set

# In[76]:


from sklearn.model_selection import train_test_split,GridSearchCV
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=31)


# # Standard Scaling

# Standardisation is a scaling technique where the values are centred around mean with a unit standard deviation and mean of attributed becomes zero.
# 
# Normalisation is good to use when we do not have gaussian distribution but generally normalization is used for models like knn and neural network.In general we use standadisation since it is not affected by outliers and it works better compared to normalisation.At the end of the day we use scaling technique which works better.
# 
# It is good practice to fit scaler on training data and then use it to transform testing data to avoid data leakage 

# In[77]:



from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train[['Item_Weight','Item_MRP','Item_Visibility','log_visibility','Calorie_Count_per_givenwt','log_Calorie']]=scaler.fit_transform(X_train[['Item_Weight','Item_MRP','Item_Visibility','log_visibility','Calorie_Count_per_givenwt','log_Calorie']])
X_test[['Item_Weight','Item_MRP','Item_Visibility','log_visibility','Calorie_Count_per_givenwt','log_Calorie']]=scaler.transform(X_test[['Item_Weight','Item_MRP','Item_Visibility','log_visibility','Calorie_Count_per_givenwt','log_Calorie']])


# In[78]:


#Printing the list of inter correlated features
correlated_features=set()
corr_matrix=df.corr()
for i in range(len(corr_matrix.columns)):
    for j in range (i):
        if(abs(corr_matrix.iloc[i,j]))>0.8:
            correlated_features.add(corr_matrix.columns[i])
correlated_features


# ## Filter Methods

# Pearson’s Correlation: It is used as a measure for quantifying linear dependence between two continuous variables X and Y. Its value varies from -1 to +1. Pearson’s correlation is given as:
# fs2
# 
# LDA: Linear discriminant analysis is used to find a linear combination of features that characterizes or separates two or more classes (or levels) of a categorical variable.
# 
# ANOVA: ANOVA stands for Analysis of variance. It is similar to LDA except for the fact that it is operated using one or more categorical independent features and one continuous dependent feature. It provides a statistical test of whether the means of several groups are equal or not.
# 
# Chi-Square: It is a is a statistical test applied to the groups of categorical features to evaluate the likelihood of correlation or association between them using their frequency distribution.
# 
# One thing that should be kept in mind is that filter methods do not remove multicollinearity. So, you must deal with multicollinearity of features as well before training models for your data.

# #### Pearson Correlation method for feature selection    (Input and Output Variable - Continuous)

# In[79]:


df.corr()


# So only Item MRP have pearson correlation value >0.5

# #### Anova test (Input Variable - Categorical Variable , Output - Continuous)

# In[80]:


from sklearn.feature_selection import chi2,f_regression
sel=f_regression(x,y)
p=pd.Series(sel[1])
p.index=x.columns
p=p[p<0.05]
p


# In[81]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
sel=SelectFromModel(LinearRegression())
sel.fit(X_train,y_train)
X_train.columns[sel.get_support()]


# ### Wraper Methods

# Forward Selection: Forward selection is an iterative method in which we start with having no feature in the model. In each iteration, we keep adding the feature which best improves our model till an addition of a new variable does not improve the performance of the model.
# 
# Backward Elimination: In backward elimination, we start with all the features and removes the least significant feature at each iteration which improves the performance of the model. We repeat this until no improvement is observed on removal of features.
# 
# Recursive Feature elimination: It is a greedy optimization algorithm which aims to find the best performing feature subset. It repeatedly creates models and keeps aside the best or the worst performing feature at each iteration. It constructs the next model with the left features until all the features are exhausted. It then ranks the features based on the order of their elimination.

# #### Multivariate Feature Selection

# In[82]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor
sfs=SFS( RandomForestRegressor(),
        k_features=5,
        forward= True,
        floating=False,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2,
        cv=None
       
       ).fit(X_train,y_train)
sfs.k_feature_names_


# In[83]:


X_train1=X_train.drop(['Calorie_Count_per_givenwt',
 'Outlet_Existence',
 'Outlet_Status',
 'log_Calorie',
 'log_visibility'],axis=1)


# In[84]:


from sklearn.feature_selection import RFE
sel=RFE(RandomForestRegressor(random_state=0),n_features_to_select=5)
sel.fit(X_train1,y_train)
X_train1.columns[sel.get_support()]


# # Training The Model

# 1 ) 'Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Identifier','Outlet_Status','Outlet_Size', 'Outlet_Location_Type','Outlet_Type'
# 
# 2 )  'Item_Weight','Item_Visibility', 'Item_MRP','Outlet_Establishment_Year','Outlet_Type'
# 
# 3 )    'Item_Weight','Item_Visibility', 'Item_MRP','Outlet_Existence','Outlet_Type'
# 
# 4 )  'Item_MRP','Outlet_Size','Outlet_Type','Outlet_Location_Type','Outlet_Status'
# 
# 5 )  'Item_MRP','Outlet_Size','Outlet_Type','Outlet_Location_Type','Outlet_Establishment_Year'
# 
# 6 )   'Item_MRP','Item_Weight','Outlet_Type','log_Calorie','Outlet_Establishment_Year','log_visibility'
# 
# 7 )    'Calorie_Count_per_givenwt','Item_Visibility', 'Item_MRP','Outlet_Existence','Outlet_Type'

# In[86]:


ya=y_train['Item_Outlet_Sales']
yb=y_test['Item_Outlet_Sales']
Xa=X_train[['Calorie_Count_per_givenwt','Item_Visibility', 'Item_MRP','Outlet_Existence','Outlet_Type']]
Xb=X_test[[ 'Calorie_Count_per_givenwt','Item_Visibility', 'Item_MRP','Outlet_Existence','Outlet_Type']]


# In[87]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
logreg=LinearRegression()
from sklearn.linear_model import Ridge
r=Ridge()
from sklearn.linear_model import Lasso
l=Lasso()
from sklearn.tree import DecisionTreeRegressor
dtc=DecisionTreeRegressor(random_state=0)
from sklearn.ensemble import RandomForestRegressor
log=RandomForestRegressor(random_state=5)
from sklearn.ensemble import GradientBoostingRegressor
gbc=GradientBoostingRegressor(random_state=0,learning_rate=0.07,max_leaf_nodes=4)
from sklearn.ensemble import BaggingRegressor
bc=BaggingRegressor(GradientBoostingRegressor(random_state=0,learning_rate=0.07,max_leaf_nodes=4))


# In[88]:


li2=['Linear','Ridge','Lasso','DecisionTree','RandomForest','Gradient','Bagging']


# In[89]:


row=[]
for j in li2:
    if j=='Linear':
        logreg.fit(Xa,ya)
        final=logreg.predict(Xb)
        row.append([j,np.sqrt(metrics.mean_squared_error(yb,final))])
    elif j=='Ridge':
        r.fit(Xa,ya)
        final1=r.predict(Xb)
        row.append([j,np.sqrt(metrics.mean_squared_error(yb,final1))])
    elif j=='Lasso':
        l.fit(Xa,ya)
        final2=l.predict(Xb)
        row.append([j,np.sqrt(metrics.mean_squared_error(yb,final2))])
    elif j=='DecisionTree':
        dtc.fit(Xa,ya)
        final3=dtc.predict(Xb)
        row.append([j,np.sqrt(metrics.mean_squared_error(yb,final3))])
    elif j=='RandomForest':
        log.fit(Xa,ya)
        final4=log.predict(Xb)
        row.append([j,np.sqrt(metrics.mean_squared_error(yb,final4))])
    elif j=='Gradient':
        gbc.fit(Xa,ya)
        final6=gbc.predict(Xb)
        row.append([j,np.sqrt(metrics.mean_squared_error(yb,final6))])
    elif j=='Bagging':
        bc.fit(Xa,ya)
        final5=bc.predict(Xb)
        row.append([j,np.sqrt(metrics.mean_squared_error(yb,final5))])


# In[90]:


row


# ### Linear Regression

# In[91]:


logreg=LinearRegression()
logreg.fit(Xa,ya)
pred=logreg.predict(Xb)
print(np.sqrt(metrics.mean_squared_error(yb,pred)))


# In[92]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(logreg,Xa,ya,cv=10,scoring='neg_mean_squared_error')
score.mean()


# ### Ridge Regression

# In[93]:


r=Ridge()
r.fit(Xa,ya)
pred0=r.predict(Xb)
print(np.sqrt(metrics.mean_squared_error(yb,pred0)))


# ### Lasso Regression

# In[94]:


l=Lasso()
l.fit(Xa,ya)
pred1=l.predict(Xb)
print(np.sqrt(metrics.mean_squared_error(yb,pred1)))


# ### Decision Tree Regressor

# In[95]:


dtc.fit(Xa,ya)
pred2=dtc.predict(Xb)
print(np.sqrt(metrics.mean_squared_error(yb,pred2)))


# In[96]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(dtc,Xa,ya,cv=10,scoring='neg_mean_squared_error')
score.mean()


# ### Random Forest Regressor

# In[101]:


from sklearn.ensemble import RandomForestRegressor
log=RandomForestRegressor(random_state=0,max_features=4)
log.fit(Xa,ya)
pred3=log.predict(Xb)
print(np.sqrt(metrics.mean_squared_error(yb,pred3)))


# In[102]:


max_features_range=np.arange(1,4,1)
n_estimators_range=np.arange(10,210,10)
param_grid = {'criterion':['mse','mae'] ,'max_features':max_features_range}
grid=GridSearchCV(log,param_grid,cv=None,scoring='neg_mean_squared_error',n_jobs=-1)
grid.fit(Xa,ya)
print(grid.best_score_)
print(grid.best_params_)


# ### Gradient Boosting Regressor

# We tried and tested 3 methods :
# 
# 1 ) Testing different hyperparameters( individual ) inside GradientBoostingRegressor
# 
# 2 ) Using GridSearchCv on hyperparameters that influence GradientBoostingRegressor, the most
# 
# 3 ) Bagging with base estimator as GradientBoostingRegressor ,and eventually , trying different bagging hyperparameters on the same
# 
# Method 1
# 
# Below are the graphs for 6 different hyperparameters of GradientBoostingRegressor
# 
# 1 ) Learning Rate
# 
# 2 ) N Estimators
# 
# 3 ) Max Depth
# 
# 4 ) Min Samples Leaf
# 
# 5 ) Min Samples Split
# 
# 6 ) Max Feature
# 
# These graphs will basically help us to look at the training and testing error , at the same time , for each and every hyperparameter
# 
# Learning Rate : - Training vs Testing Error

# In[102]:


from sklearn.ensemble import GradientBoostingRegressor
gbc=GradientBoostingRegressor(max_leaf_nodes=6,min_samples_leaf=5)
gbc.fit(Xa,ya)
pred4=gbc.predict(Xb)
pp=gbc.predict(Xa)
print(np.sqrt(metrics.mean_squared_error(yb,pred4)))
print(np.sqrt(metrics.mean_squared_error(ya,pp)))


# ##### Hyperparameter tuning by checking training and testing error
# 

# In[103]:


min_samples_leaf=[0.1,0.2,0.3,0.4,0.5,1,2,3,4,5]
train_result=[]
test_result=[]
for i in min_samples_leaf:
    gbc=GradientBoostingRegressor( random_state=0,min_samples_leaf=i)
    gbc.fit(Xa,ya)
    pred4=gbc.predict(Xb)
    error1=np.sqrt(metrics.mean_squared_error(yb,pred4))
    test_result.append(error1)
    pre=gbc.predict(Xa)
    error2=np.sqrt(metrics.mean_squared_error(ya,pre))
    train_result.append(error2)
plt.plot(min_samples_leaf,train_result)
plt.plot(min_samples_leaf,test_result)
plt.show()


# In[104]:


max_features=[1,2,3,4]
train_result=[]
test_result=[]
for i in max_features:
    gbc=GradientBoostingRegressor( random_state=0,max_features=i)
    gbc.fit(Xa,ya)
    pred4=gbc.predict(Xb)
    error3=np.sqrt(metrics.mean_squared_error(yb,pred4))
    test_result.append(error3)
    pre=gbc.predict(Xa)
    error4=np.sqrt(metrics.mean_squared_error(ya,pre))
    train_result.append(error4)
plt.plot(max_features,train_result,label='train')
plt.plot(max_features,test_result,label='test')
plt.legend(loc='upper left')
plt.show()


# In[105]:


min_samples_split=[0.1,0.2,0.3,0.4]
train_result=[]
test_result=[]
for i in min_samples_split:
    gbc=GradientBoostingRegressor( random_state=0,min_samples_split=i)
    gbc.fit(Xa,ya)
    pred4=gbc.predict(Xb)
    error3=np.sqrt(metrics.mean_squared_error(yb,pred4))
    test_result.append(error3)
    pre=gbc.predict(Xa)
    error4=np.sqrt(metrics.mean_squared_error(ya,pre))
    train_result.append(error4)
plt.plot(min_samples_split,train_result,label='train')
plt.plot(min_samples_split,test_result,label='test')
plt.legend(loc='upper left')
plt.show()


# In[106]:


max_depth=[1,2,3,4]
train_result=[]
test_result=[]
for i in max_depth:
    gbc=GradientBoostingRegressor( random_state=0,max_depth=i)
    gbc.fit(Xa,ya)
    pred4=gbc.predict(Xb)
    error3=np.sqrt(metrics.mean_squared_error(yb,pred4))
    test_result.append(error3)
    pre=gbc.predict(Xa)
    error4=np.sqrt(metrics.mean_squared_error(ya,pre))
    train_result.append(error4)
plt.plot(max_depth,train_result,label='train')
plt.plot(max_depth,test_result,label='test')
plt.legend(loc='upper left')
plt.show()


# In[107]:


max_leaf_nodes=[2,3,4,5]
train_result=[]
test_result=[]
for i in max_leaf_nodes:
    gbc=GradientBoostingRegressor( random_state=0,max_leaf_nodes=i)
    gbc.fit(Xa,ya)
    pred4=gbc.predict(Xb)
    error3=np.sqrt(metrics.mean_squared_error(yb,pred4))
    test_result.append(error3)
    pre=gbc.predict(Xa)
    error4=np.sqrt(metrics.mean_squared_error(ya,pre))
    train_result.append(error4)
plt.plot(max_leaf_nodes,train_result,label='train')
plt.plot(max_leaf_nodes,test_result,label='test')
plt.legend(loc='upper left')
plt.show()


# In[108]:


learning_rate=[0.01,0.1,0.2,0.3,0.4]
train_result=[]
test_result=[]
for i in learning_rate:
    gbc=GradientBoostingRegressor( random_state=0,learning_rate=i)
    gbc.fit(Xa,ya)
    pred4=gbc.predict(Xb)
    error3=np.sqrt(metrics.mean_squared_error(yb,pred4))
    test_result.append(error3)
    pre=gbc.predict(Xa)
    error4=np.sqrt(metrics.mean_squared_error(ya,pre))
    train_result.append(error4)
plt.plot(learning_rate,train_result,label='train')
plt.plot(learning_rate,test_result,label='test')
plt.legend(loc='upper left')
plt.show()


# In[109]:


random_state=[1,2,3,4,5,6,7]
train_result=[]
test_result=[]
for i in random_state:
    gbc=GradientBoostingRegressor(random_state=i)
    gbc.fit(Xa,ya)
    pred4=gbc.predict(Xb)
    error3=np.sqrt(metrics.mean_squared_error(yb,pred4))
    test_result.append(error3)
    pre=gbc.predict(Xa)
    error4=np.sqrt(metrics.mean_squared_error(ya,pre))
    train_result.append(error4)
plt.plot(random_state,train_result,label='train')
plt.plot(random_state,test_result,label='test')
plt.legend(loc='upper left')
plt.show()


# max Depth =3
# 
# min sample split =0.2
# 
# Max Features=4
# 
# Min Sample leaf=0.1
# 
# Max leaf nodes=4
# 
# learning rate= 0.07

# ##### Hyperparameter tuning using GridSearchCV

# In[110]:


max_features_range=np.arange(1,6,1)
min_sample_range=[0.1,0.2,0.3,1,2,3,4,5]
param_grid = {'max_features':max_features_range,'min_samples_leaf':min_sample_range,'max_leaf_nodes':[2,3,4,5,6],'min_samples_split':[0.1,0.2,0.3,0.4]}
grid=GridSearchCV(gbc,param_grid,cv=None,scoring='neg_mean_squared_error',n_jobs=-1)
grid.fit(Xa,ya)
print(grid.best_score_)
print(grid.best_params_)


# ### Ada Boost Regressor

# In[111]:


from sklearn.ensemble import AdaBoostRegressor
abc=AdaBoostRegressor(random_state=0,n_estimators=10,learning_rate=0.1)
abc.fit(Xa,ya)
pred5=abc.predict(Xb)
print(np.sqrt(metrics.mean_squared_error(yb,pred5)))


# ### Bagging Regressor

# In[112]:


from sklearn.ensemble import BaggingRegressor
bc=BaggingRegressor(GradientBoostingRegressor(random_state=0),random_state=4,n_estimators=5)
bc.fit(Xa,ya)
pred6=bc.predict(Xb)
ppp=bc.predict(Xa)
print(np.sqrt(metrics.mean_squared_error(yb,pred6)))
print(np.sqrt(metrics.mean_squared_error(ya,ppp)))


# In[113]:


random_state=[1,2,3,4,5,6,7]
train_result=[]
test_result=[]
for i in random_state:
    bc=BaggingRegressor(GradientBoostingRegressor(random_state=0),random_state=i)
    bc.fit(Xa,ya)
    pred4=bc.predict(Xb)
    error3=np.sqrt(metrics.mean_squared_error(yb,pred4))
    test_result.append(error3)
    pre=bc.predict(Xa)
    error4=np.sqrt(metrics.mean_squared_error(ya,pre))
    train_result.append(error4)
plt.plot(random_state,train_result,label='train')
plt.plot(random_state,test_result,label='test')
plt.legend(loc='upper left')
plt.show()


# ### Stacking Regressor

# In[100]:


from mlxtend.regressor import StackingRegressor
scc=StackingRegressor(regressors=[gbc],meta_regressor=bc)
scc.fit(Xa,ya)
pred7=scc.predict(Xb)
print(np.sqrt(metrics.mean_squared_error(yb,pred7)))


# # Preparing testing dataset for final submission

# We shall now deal with testing file, in the same manner as we dealt with our training file . Except , for the fact that we wont remove Outliers , from our testing file ; in order to maintain the integrity of the data .
# 
# You might skip the next few cells , until the title 'Finally!!!'

# #### Filling the missing value

# In[91]:


df_test['Item_Fat_Content'].replace('LF','Low Fat',inplace=True)
df_test['Item_Fat_Content'].replace('low fat','Low Fat',inplace=True)
df_test['Item_Fat_Content'].replace('reg','Regular',inplace=True)


# In[92]:


df_test['Item_Visibility'].replace(0,np.nan,inplace=True)


# In[93]:


df_test.isnull().sum()


# In[94]:


df_test['Outlet_Size'].fillna('Small',inplace=True)


# In[95]:


df_test['Outlet_Type1']=df_test['Outlet_Type']
dict={'Supermarket Type1':'Supermarket','Supermarket Type2':'Supermarket','Supermarket Type3':'Supermarket','Grocery Store':'Grocery Store'}
df_test['Outlet_Type1']=df_test['Outlet_Type1'].map(dict)
df_test.isnull().sum()


# In[96]:


df0=df_test[df_test['Item_Identifier']=='FDU28']
df0


# In[97]:


df_test['Item_Visibility'].fillna(df_test.groupby(['Item_Identifier','Outlet_Type1'])['Item_Visibility'].transform('mean'),inplace=True)


# In[98]:


df_test['Item_Visibility']=(df_test.groupby('Item_Identifier')['Item_Visibility'].transform(lambda x:x.fillna(x.mean()+0.1)))


# In[99]:


a=df_test[df_test['Outlet_Type1']=='Supermarket']
m=a['Item_Visibility'].mean()
print(m)
b=df_test[df_test['Outlet_Type1']=='Grocery Store']
m1=b['Item_Visibility'].mean()
print(m1)


# In[100]:


dict={'Supermarket':0.06416203204468011,'Grocery Store':0.11417913043057272}
df_test['Item_Visibility'].fillna(df_test['Outlet_Type1'].map(dict),inplace=True)


# In[101]:


df_test['Item_Weight'].fillna(df_test.groupby('Item_Identifier')['Item_Weight'].transform('median'),inplace=True)
df_test['Item_Weight'].fillna(df_test.groupby('Item_Type')['Item_Weight'].transform('mean'),inplace=True)


# In[102]:


df_test.isnull().sum()


# In[103]:


df_test['log_visibility']=np.log(df_test['Item_Visibility'])


# #### Feature Engineering

# In[104]:


df_test['MRP_bin']=pd.cut(df_test['Item_MRP'],bins=[31,94.413,141.4155,186.0267,267],labels=['cheap','affordable','slightly expensive','expensive'])
df_test['Weight_bin']=pd.cut(df_test['Item_Weight'],bins=[4,8.64,12.36,16.71,22],labels=['vlight','light','moderate','heavy'])


# In[105]:


df_test['Item_Identifier_temp']=df_test['Item_Identifier'].str[:2]
dict={'Supermarket Type1':'Supermarket Type1','Supermarket Type2':'Supermarket Type2','Supermarket Type3':'Supermarket Type2','Grocery Store':'Grocery Store'}
df_test['Outlet_Type']=df_test['Outlet_Type'].map(dict)
df_test.head()


# In[106]:


df_test['Outlet_Existence']=[2020-i for i in df_test['Outlet_Establishment_Year']]
df_test['Outlet_Status']=[0 if i<=2000 else 1 for i in df_test['Outlet_Establishment_Year']]
df_test.loc[df_test['Item_Identifier_temp']=='NC','Item_Fat_Content']='No Fat'


# In[107]:


# Craeted a new column named calorie count which will show the calorie of corresponding item type
def set_cal(df_test):
    if df_test['Item_Type']=='Dairy':
        return 46
    elif df_test['Item_Type']=='Soft Drinks':
        return 51
    elif df_test['Item_Type']=='Meat':
        return 143
    elif df_test['Item_Type']=='Fruits and Vegetables':
        return 65
    elif df_test['Item_Type']=='Baking Goods':
        return 140
    elif df_test['Item_Type']=='Snack Foods':
        return 475
    elif df_test['Item_Type']=='Frozen Foods':
        return 50
    elif df_test['Item_Type']=='Breakfast':
        return 350
    elif df_test['Item_Type']=='Hard Drinks':
        return 250
    elif df_test['Item_Type']=='Canned':
        return 80
    elif df_test['Item_Type']=='Starchy Foods':
        return 90
    elif df_test['Item_Type']=='Seafood':
        return 204
    elif df_test['Item_Type']=='Breads':
        return 250
    else:
        return 0
df_test['Calorie_Count_per_100g']=df_test.apply(set_cal,axis=1)
df_test['Calorie_Count_per_givenwt']=df_test['Calorie_Count_per_100g']/100
df_test['Calorie_Count_per_givenwt']=df_test['Calorie_Count_per_givenwt']*df_test['Item_Weight']
df_test['log_Calorie']=np.sqrt(df_test['Calorie_Count_per_givenwt'])


# In[108]:


dict={'Dairy':'D','Meat':'S','Fruits and Vegetables':'D','Breakfast':'S','Breads':'S','Starchy Foods':'S','Seafood':'S','Soft Drinks':'S','Household':'D','Baking Goods':'S','Snack Foods':'D','Frozen Foods':'D','Hard Drinks':'S','Canned':'D','Health and Hygiene':'S','Others':'S'}
df_test['Item_Frequency']=df_test['Item_Type'].map(dict)


# #### Label Encoding

# In[109]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_test['Item_Identifier_temp']=le.fit_transform(df_test['Item_Identifier_temp'])
df_test['Item_Type']=le.fit_transform(df_test['Item_Type'])
df_test['Outlet_Type']=le.fit_transform(df_test['Outlet_Type'])
df_test['Outlet_Identifier']=le.fit_transform(df_test['Outlet_Identifier'])
df_test['Outlet_Type1']=le.fit_transform(df_test['Outlet_Type1'])
dict={'Small':0,'Medium':1,'High':2}
df_test['Outlet_Size']=df_test['Outlet_Size'].map(dict)
dict1={'cheap':0,'affordable':1,'slightly expensive':2,'expensive':3}
df_test['MRP_bin']=df_test['MRP_bin'].map(dict1)
dict2={'vlight':0,'light':1,'moderate':2,'heavy':3}
df_test['Weight_bin']=df_test['Weight_bin'].map(dict2)
dict3={'Tier 3':0,'Tier 2':1,'Tier 1':2}
df_test['Outlet_Location_Type']=df_test['Outlet_Location_Type'].map(dict3)
dict4={'No Fat':0,'Low Fat':1,'Regular':2}
df_test['Item_Fat_Content']=df_test['Item_Fat_Content'].map(dict4)
dict5={35:0,33:1,23:2,22:3,21:4,18:5,16:6,13:7,11:8}
df_test['Outlet_Existence']=df_test['Outlet_Existence'].map(dict5)
dict6={'D':1,'S':0}
df_test['Item_Frequency']=df_test['Item_Frequency'].map(dict6)
dict7={1985:0,1987:1,1997:2,1998:3,1999:4,2002:5,2004:6,2007:7,2009:8}
df_test['Outlet_Establishment_Year']=df_test['Outlet_Establishment_Year'].map(dict7)


# In[110]:


df_test.head()


# In[111]:


df_test.describe()


# #### Making the final prediction

# In[112]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_test[['Item_Weight','Item_MRP','Item_Visibility','log_visibility','Calorie_Count_per_givenwt','log_Calorie']]=scaler.fit_transform(df_test[['Item_Weight','Item_MRP','Item_Visibility','log_visibility','Calorie_Count_per_givenwt','log_Calorie']])
df_test.head()


# In[113]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df[['Item_Weight','Item_MRP','Item_Visibility','log_visibility','Calorie_Count_per_givenwt','log_Calorie']]=scaler.fit_transform(df[['Item_Weight','Item_MRP','Item_Visibility','log_visibility','Calorie_Count_per_givenwt','log_Calorie']])
df.head()


# In[229]:


X1=df[[ 'Calorie_Count_per_givenwt','Item_Visibility', 'Item_MRP','Outlet_Existence','Outlet_Type']]
y1=df['Item_Outlet_Sales']


# In[230]:


XX=df_test[[ 'Calorie_Count_per_givenwt','Item_Visibility', 'Item_MRP','Outlet_Existence','Outlet_Type']]


# # Finally this is our best model and best set of hyperparameters

# In[231]:


from sklearn.ensemble import BaggingRegressor
bc=BaggingRegressor(GradientBoostingRegressor(random_state=7),random_state=4,n_estimators=5)
bc.fit(X1,y1)
predd=bc.predict(XX)
predd=np.where(predd>0,predd,100)
predd=pd.DataFrame(predd,columns=['Item_Outlet_Sales'])


# In[233]:


predd.to_csv('C:\\Users\\User-1\\Desktop\\final2.csv')

