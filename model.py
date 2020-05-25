#!/usr/bin/env python
# coding: utf-8

# <h1><center>Car Emissions</center></h1>
# 
# This dataset contains car emissions data for UK, which includes over 5,000 vehicles and their Emissions of August 2017.
# 
# ![sales](https://cdn.britannica.com/76/155676-050-40CF909F/Air-pollution-vehicle-tailpipes-number-criteria-pollutants.jpg)
# 

# # cars emissions data considered as vehicular pollution
# 
# # Our Goal in Steps:
# ## 1- Read the data and fetch it include:
#    ###   - Fetch the head of data
#    ###   - Discribe the data
#    ###   - See more info about it
#    ###   - See the types of the data
# 
# ## 2- Clean and Preprocess on the dataset include:
#    ###   - Drop the rows only if all of the values in the row are missing:
#    ###   - See how many nulls we have in the columns:
#    ###   - Drop uneeded columns that will not effect the emission rate
#    ###   - Drop clomuns with nulls more than 60% but if we need some of it we will do a replacements
#    ###   - Replace some nulls values with means or common values
#    ###   - Enode some catagorial features if we need 
#    ###   - Drop the rows only if all of the values in the row are missing:
# 
# ## 3- what insights can we get from it ? using machine learrning ^_^ includes:
#    ###   - Define which is your prediction target
#    ###   - Define the train and test datasets
#    ###   - Evaluate the R square
#    ###   - Choose the ML algorthim
#    ###   - Compare predtion results with the test dataset

# #                                           STEP 1

# # Read the data and fetch

# In[1]:


# import first
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# change the style from the very beging
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
# Installing and loading the library
#!pip install dabl

import dabl


# In[2]:


emission = pd.read_csv('Car_emission.csv', encoding='windows-1252')
emission


# In[3]:


emission.info()


# In[4]:


emission.dtypes


# In[5]:


emission.describe()


# # STEP 2

# #EDA
# Let's create some simple plots to check out the data!

# In[6]:


sns.heatmap(emission.corr())


# # Cleaning and preprocessing:

# ### First we Will drop the rows only if all of the values in the row are missing:

# In[7]:


emission.dropna(how = 'all',inplace = True)


# In[8]:


mis_val = emission.isnull().sum()


# In[9]:


# Missing values
def missing_values_table(emission):
        # Total missing values
        mis_val = emission.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * emission.isnull().sum() / len(emission)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(emission.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

missing_values= missing_values_table(emission)
missing_values.style.background_gradient(cmap='Reds')


# In[10]:


# installation and importing the library
#!pip install missingno
import missingno as msno


# In[11]:


msno.bar(emission)
#displaying the missing_values with the help of barplot 


# In[12]:


msno.heatmap(emission)


# In[13]:


sns.heatmap(emission.isnull(), cbar=True)


# ## Handling the missing data
# ### Drop uneeded colmuns with full null values:

# In[14]:


clean_dataset = emission
clean_dataset = clean_dataset.drop(['Unnamed: 27','Unnamed: 28','Unnamed: 29','Unnamed: 30','Unnamed: 31','Unnamed: 32',
                                   'Unnamed: 33'], axis=1)


# In[15]:


# but we still need to hundle the rest nulls values by droping or replace:
clean_dataset.isna().sum()


# In[16]:


clean_dataset.head()


# In[17]:


clean_dataset = clean_dataset.drop(['Manufacturer','Model'], axis=1)


# In[18]:


clean_dataset = clean_dataset.drop(['Description'], axis=1)


# ### Drop uneeded colmuns with full null values more than 60%:

# In[19]:


clean_dataset = clean_dataset.loc[:, clean_dataset.isin([' ','NULL',0]).mean() < .6]
print (clean_dataset)


# In[20]:


clean_dataset = clean_dataset.drop(['Electric energy consumption Miles/kWh',
                                    'wh/km',
                                    'Maximum range (Km)'
                                    ,'Maximum range (Miles)',
                                    'Electricity cost',
                                    'THC Emissions [mg/km]'
                                    ,'THC + NOx Emissions [mg/km]',
                                    'Particulates [No.] [mg/km]'], axis=1)


# ### now we will see if there is dupilcation and remove them if we need it:

# ## as you can see below we dont need to remove any of them becuase cost can be duplicated in our case

# In[21]:


clean_dataset.describe(include=['O'])


# In[22]:


clean_dataset.head()


# In[23]:


clean_dataset.info()


# ### now for numirc nan values we will replace the needed columns with the mean values of there respictive columns  like manufacture , model and Description we dont need them for emission:

# In[24]:


clean_dataset.iloc[:,[10,11]]


# In[25]:


#convert object to string and remove the euro sign
clean_dataset['Fuel Cost 12000 Miles'] = clean_dataset['Fuel Cost 12000 Miles'].str.slice(1,len(clean_dataset['Fuel Cost 12000 Miles'])-1)
#convert the string with ',' to float first remove ',' then convert to float
clean_dataset['Fuel Cost 12000 Miles'] = clean_dataset['Fuel Cost 12000 Miles'].str.replace(",","").astype(float)


# In[26]:


clean_dataset.iloc[:,[10,11]]


# ### Do the same to Total cost / 12000 miles remove Euro sign then remove the ',' then convert it to float

# In[27]:


#convert object to string and remove the euro sign
clean_dataset['Total cost / 12000 miles'] = clean_dataset['Total cost / 12000 miles'].str.slice(1,len(clean_dataset['Total cost / 12000 miles'])-1)
#convert the string with ',' to float first remove ',' then convert to float
clean_dataset['Total cost / 12000 miles'] = clean_dataset['Total cost / 12000 miles'].str.replace(",","").astype(float)


# In[28]:


clean_dataset.iloc[:,[10,11]]


# In[29]:


# convert cost objects to numirc
# Fuel Cost 12000 Miles                    
# Electricity cost                        
# Total cost / 12000 miles
#check nulls numirc
# but we still need to hundle the rest nulls values by droping or replace:
clean_dataset.isna().sum()


# ## replace it by the mean (which is what we're going to do)
# ## Let's see how to replace them with sklearn

# In[30]:


#importing sklrn
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
clean_dataset[['Fuel Cost 12000 Miles', 
                 'Emissions CO [mg/km]',
                 'Engine Capacity',
                 'Metric Urban (Cold)',
                 'Metric Extra-Urban',
                 'Metric Combined',
                'Imperial Urban (Cold)',
                'Imperial Extra-Urban',
                'Imperial Combined',
                'Emissions CO [mg/km]',
                'Emissions NOx [mg/km]']]= imputer.fit_transform(clean_dataset[[
                                         'Fuel Cost 12000 Miles', 
                                         'Emissions CO [mg/km]',
                                         'Engine Capacity',
                                         'Metric Urban (Cold)',
                                         'Metric Extra-Urban',
                                         'Metric Combined',
                                        'Imperial Urban (Cold)',
                                        'Imperial Extra-Urban',
                                        'Imperial Combined',
                                        'Emissions CO [mg/km]',
                                        'Emissions NOx [mg/km]']])


# ### check nulls again:

# In[31]:


clean_dataset.isna().sum()


# ## now we have the last nan values from Transmission by filling them with the common value of the column:

# In[32]:


clean_dataset['Transmission'].fillna(method='ffill', inplace=True)


# ### check nulls again:

# In[33]:


clean_dataset.isna().sum()


# ## Now for final Cleanning step:
# ### Note the Fuel Type  and Transmission are catgorial feature we will   ENCODE Them:

# In[34]:


# import the needed librarys
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
clean_dataset['Transmission'] = encoder.fit_transform(clean_dataset['Transmission'])


# In[35]:


# import the needed librarys
encoder = LabelEncoder()
clean_dataset['Fuel Type'] = encoder.fit_transform(clean_dataset['Fuel Type'])


# # ------------------------------------------------STEP 3----------------------------------------------------

# # Congrats We have Finished Cleanning & Preproccesing Now For the Final Setp 'Prediction of Emissions CO [mg/km] ':

# ## Let's see the correlation via a heat map !

# In[36]:


corr = clean_dataset.corr()
fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
plt.yticks(range(len(corr.columns)), corr.columns)
plt.grid(True)
plt.show()


# ## Implementing Regression:

# In[37]:


import seaborn as sns
sns.heatmap(corr)


# ## now we need to seperate the dependant and independent variables

# In[38]:


clean_dataset


# In[39]:


#features_matrix = clean_dataset.drop('Emissions CO [mg/km]', axis=1)
features_matrix = clean_dataset[['Imperial Extra-Urban']]
goal_vector = clean_dataset['Emissions CO [mg/km]']


# ### We replaced the data with numbers that do the same job, as for the model it doesn't matter if it's called AMT6 or 1 for example yet, the model might think that some Transmission has larger value than the other, and this might cause us some mistaken calculations ! a good way to handle such case, is to use the one hot encoder Now let's have our data encoded in a dummy variables, so we no longer have the problem of one country or(category) taking a higher value than the other
# 

# In[40]:


# # import the oneHotEncoder class
# from sklearn.preprocessing import OneHotEncoder
# oneHotEncoder = OneHotEncoder(categorical_features=[0])
# features_matrix = oneHotEncoder.fit_transform(features_matrix)
# features_matrix


# # Training a Linear Regression Model

# In[41]:


# import the modules 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features_matrix, goal_vector, test_size = 0.2, random_state = 0)


# In[42]:


x_train


# In[43]:


#fitting mulitiple linear regrission model to TRAINNING set:
from sklearn.linear_model import LinearRegression
#create opbject of linear regression model
Regression= LinearRegression()
Regression.fit(x_train,y_train)


# In[44]:


# predicting the test set result
y_pred = Regression.predict(x_test)
y_pred


# ### Some Extra Information

# In[45]:


#calculate coeffieant
print(Regression.coef_)


# In[46]:


#calculate the interpet
print(Regression.intercept_)


# ## R Square Value

# In[47]:


#evaluating R sequare value
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)#r2_score is getting the diffrence between y_test(the acual result) and y_predict


# # Our first simple model of Linear Regression of Emission CO prediction

# In[48]:


compare = pd.DataFrame({'Prediction': y_pred, 'Test Data' : y_test})
compare.head(10)


# In[49]:


plt.plot(x_test, y_test,'.', x_test, y_pred, '-')
plt.title('our first simple model of Linear Regression of Emission CO prediction:')
plt.xlabel('Feature')
plt.ylabel('Emission')
plt.show()


# In[50]:


import pickle


# In[51]:


# Saving model to disk
pickle.dump(Regression, open('Regression.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('Regression.pkl','rb'))
print(model.predict([[57]]))


# In[ ]:




