# -*- coding: utf-8 -*-
#It was previously an .ipynb file, I'd suggest converting it before running.
"""CRM_segment.ipynb
"""

#Importing libraries
import pandas as pd
import numpy as np
import dateutil
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mat
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

#Pandas commands to avoid warnings and unlock visualization of all the columns of any df
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = None

#Reading the csv file and presenting its columns
#I've checked its values, length and nulls before, it won't be presented here
df = pd.read_csv('data.csv')
print(df.columns)

#As an RFM method is going to be applied, there's the need to create the F and M columns
# Recency is already in the df
# Frequency is the total number of transactions in the df
# Monetary is the total amount spent by the customer
df['frequency']=df['NumDealsPurchases']+df['NumWebPurchases']+df['NumCatalogPurchases']+df['NumStorePurchases']
df['monetary']=df['MntWines']+df['MntFruits']+df['MntMeatProducts']+df['MntFishProducts']+df['MntSweetProducts']+df['MntGoldProds']

#Creation of the RFM df and its description
rfm = df[['ID','Recency','frequency','monetary']]
rfm.describe()

#Ranking the clients according to the RFM method

def rank_r(x,p,t):
    if x <=t[p][0.25]:
        return str(1)
    elif x<=t[p][0.75]:
        return str(2)
    else:
        return str(3)
    
def rank_f(x,p,t):
    if x<= t[p][0.75]:
        return str(3)
    else:
        return str(1)
    
def rank_m(x,p,t):
    if x<= t[p][0.25]:
        return str(3)
    elif x<= t[p][0.75]:
        return str(2)
    else:
        return str(1)

#This function defines which segment the customer belongs to, according to his RFM score  
#definir_segmento stands for define_segment
def definir_segmento(rows):
    if rows['rfm_score']=='111':
        return 'best_users'
    elif rows['rfm_score']=='211':
        return 'almost_lost'
    elif rows['rfm_score']=='311':
        return 'lost_users'
    elif rows['rank_r']=='3':
        return 'cheap_lost'
    elif rows['rank_f']=='1':
        return 'loyal_users'
    elif rows['rank_m']=='1':
        return 'big_spender'
    elif rows['rank_f']=='3':
        return 'new_customer'
    else:
        return rows['rfm_score']

#The 25 and 75th quantiles are used to rank the RFM
#limiar stands for threshold
#limiar needs to be converted into a dictionary to be applied in the RFM
limiar = rfm.drop('ID', axis=1).quantile(q=[0.25, 0.75])
limiar = limiar.to_dict()

#applying the rank functions and score function
rfm['rank_r']=rfm['Recency'].apply(rank_r, args=('Recency', limiar))
rfm['rank_f']=rfm['frequency'].apply(rank_f, args=('frequency', limiar))
rfm['rank_m']=rfm['monetary'].apply(rank_m, args=('monetary',limiar))
rfm['rfm_score']=rfm['rank_r']+rfm['rank_f']+rfm['rank_m']

#storing in a column
rfm['segment']=rfm.apply(definir_segmento, axis=1)

#creating a new array and aggregating it 
rfm_count=rfm.groupby('segment').agg({'ID':['count'], 'monetary':['sum']}).reset_index()
rfm_count.columns = ['segment', 'user', 'amount']
rfm_count[['amount']]=rfm_count[['amount']]/100

#plot
fig1, ax1 = plt.subplots(figsize=(10,8))
ax1.pie(data=rfm_count,x='user', labels='segment', autopct='%1.1f%%',
        startangle=90, colors=['#a02331','#9C050B','#F7ACB7','#DA5D69','#FED298','#F5E9DA','#EA1D2C'])

ax1.axis('equal')
plt.tight_layout()
plt.title('Users percentage for each segment')
plt.show()

#plotting the percentage of spent amount per customer segment
fig1,ax1=plt.subplots(figsize=(10,8))
ax1.pie(data=rfm_count,x='amount', labels='segment', autopct='%1.1f%%', startangle=90,
       colors=['#a02331','#9C050B','#F7ACB7','#DA5D69','#FED298','#F5E9DA','#EA1D2C'])

ax1.axis('equal')

plt.tight_layout()
plt.title('Percentage of amount spent by each customer segment')
plt.show()

#the RFM method define that the high growth customers are
#best users, as they buy recently, frequently and spend lots of money
#and big spenders, as they buy not so frequently but the monetary
#value of each transaction is higher

#creating a function to create a column of high growth users:

def categorizacao(rows):
    if rows['segment'] =='best_users':
        return 1
    elif rows['segment']=='big_spender':
        return 1
    else:
        return 0
    
rfm['high_growth']=rfm.apply(categorizacao, axis=1)
categoria = rfm[['ID','high_growth']]

#analyzing the new column to see the percentage of users
#divided by the growth categorization

growth_count=rfm.groupby('high_growth').agg({'ID':['count'],'monetary':['sum']}).reset_index()
growth_count.columns=['segment','user','amount']
growth_count.loc[growth_count['segment']==0, 'segment']='low growth'
growth_count.loc[growth_count['segment']==1, 'segment']='high growth'

fig1, ax1 = plt.subplots()
ax1.pie(data=growth_count, x='user', labels='segment', autopct='%1.1f%%',
        startangle=90)
ax1.axis('equal')
plt.tight_layout()
plt.title('Percentage of users in each growth segment')
plt.show()

#also plotting their spent amount

fig1, ax1= plt.subplots()
ax1.pie(data=growth_count, x='amount', labels='segment', autopct='%1.1f%%',
        startangle=90)

ax1.axis('equal')
plt.tight_layout()
plt.title('Amount spent by growth segment')
plt.show()

#creating new columns in the initial df that may be useful for predictions
today=datetime.now()
df['Dt_Customer']=pd.to_datetime(df['Dt_Customer'], format='%Y-%m-%d')
df['year']=df['Dt_Customer'].apply(lambda x: x.year)
df['tenure']=df['Dt_Customer'].apply(lambda x: (today-x).days)

#creating a new df to use for training prediction

training_dataset=df[['ID','tenure','Recency','monetary','frequency','MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Kidhome','Teenhome', 'Income']]
training_dataset=training_dataset.merge(categoria)

#identifying columns with NA values and filling with the mean values
na_cols = training_dataset.isna().any()
na_cols = na_cols[na_cols == True].reset_index()
na_cols = na_cols['index'].tolist()

for col in training_dataset.columns[1:]:
    if col in na_cols:
        if training_dataset[col].dtype != 'object':
            training_dataset[col] = training_dataset[col].fillna(training_dataset[col].mean()).round(0)

#Function to print the results of the prediction test
def print_result(y_test,y_pred):
    target_names=['Low Growth','High Growth']
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))
    print('Confusion Matrix: \n', metrics.confusion_matrix(y_test,y_pred))

#Using logistic regression
#Splitting the data into a (30-70) ratio for testing and training
X = training_dataset.drop(columns=['ID','high_growth'])
X = preprocessing.scale(X)
y = training_dataset['high_growth']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

#Preparing the presentation of the solver
print("\n\nLogistic Regression...")
log = linear_model.LogisticRegression(solver = 'liblinear')
log.fit(X_train, y_train)

print("\n\nTraining")
y_pred = log.predict(X_train)
print_result(y_train, y_pred)

print("\n\nTesting")
#now predict on test data
y_pred = log.predict(X_test)
print_result(y_test, y_pred)

#There's a decent performance predicting Low Growth customer
#however this code aims to the High Growth ones

#Taking a look at the data correlation to understand the relationship
#between the variables of the dataset

def correlation_heatmap(training_dataset):
    correlations = training_dataset.corr()
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();

correlation_heatmap(training_dataset)

#Using cross-validation as an alternative
#default is 10-fold

X = training_dataset.drop(columns =['ID','high_growth'])
X = preprocessing.scale(X)
y = training_dataset['high_growth']


log = linear_model.LogisticRegression(solver = 'liblinear')
y_pred = cross_val_predict(log, X, y)
print("\n\nLogistic Regression...\n\n")
print_result(y,y_pred)

#The results were even worse

#As there's a big imbalance nature of the data, in high/low growth
#customer ratio, there's the possibility of oversampling the
#minority class until the ratio is balanced
#SMOTE may be used to avoid random oversampling and overfitting

sm = SMOTE(random_state=50)

X = training_dataset.drop(columns=['ID','high_growth'])
y = training_dataset['high_growth']

X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50) 
#once again, 70% training and 30% testing

#oversampling the training dataset
x_train_res, y_train_res = sm.fit_resample(X_train, y_train)
x_test_res, y_test_res = sm.fit_resample(X_test, y_test)
print (y_train.value_counts())
print("New Data", np.bincount(y_train_res))

print("\n\nLogistic Regression...")
log = linear_model.LogisticRegression(solver='liblinear')
log.fit(x_train_res, y_train_res)

#presenting the training results
print("\n\nTraining")
y_pred = log.predict(x_train_res)
print_result(y_train_res, y_pred)
#predicting on test data
print("\n\nTesting")
y_pred = log.predict(X_test)
print_result(y_test, y_pred)

#there was an overall progress with this method

#lastly, let's try Decision Tree method and Random Forest

X = training_dataset.drop(columns=['ID','high_growth'])
y = training_dataset['high_growth']

decision_tree_classifier = DecisionTreeClassifier()
y_pred_dt = cross_val_predict(decision_tree_classifier, X, y)
print("\n\nDecision Tree...")
print_result(y,y_pred_dt)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
random_forest_classifier = RandomForestClassifier(n_estimators=10, criterion="entropy")
y_pred_rf = cross_val_predict(random_forest_classifier, X, y)
print("\n\nRandom Forest...")
print_result(y, y_pred_rf)    

#As it may be seen, 100% of success on prediction may be overfitted
#or have any other kind of problem. However, 99% of precision is 
#acceptable and very good. Random forest prediction will be the
#chosen method

#saving the High Growth predicted customers in a csv file
training_dataset['predicted']=y_pred_rf
high_growth_customers = training_dataset.loc[training_dataset['predicted']==True]

high_growth_customers.sort_values('monetary',ascending=False).to_csv('high_growth_customer.csv',index=False)

#print('High Growth Customer: \n',high_growth_customers.sort_values('monetary', ascending=False))

#creating the final set of information to be analyzed
base_final = pd.merge(
    left=high_growth_customers['ID'],
    right=df,
    left_on='ID',
    right_on='ID',
    how='left')

base_final.sort_values('monetary', ascending=False)

#plotting some histograms to check if there's any kind of standard information
# about these customers
base_final2 = base_final[['Year_Birth','Income','Recency', 'MntWines',
                'MntFruits','MntMeatProducts','MntFishProducts',
                'MntSweetProducts','MntGoldProds']]

fig = plt.figure(figsize=(15,12))
plt.suptitle('Numeric columns histogram \n', horizontalalignment='center',
             fontstyle='normal', fontsize=24, fontfamily = 'verdana')

for i in range(base_final2.shape[1]):
    plt.subplot(6,3,i+1)
    f=plt.gca()
    f.set_title(base_final2.columns.values[i])
    
    vals = np.size(base_final2.iloc[:,i].unique())
    if vals >=100:
        vals=100
    
    plt.hist(base_final2.iloc[:,i], bins=vals, color ='#ea1d2c')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

base_final.describe()

#presenting these customers splitted by education level
#there's a pattern here, most part of them are graduated or have higher education
education_split=base_final[['ID','Education']]
sectors = education_split.groupby('Education')
education_split = pd.DataFrame(sectors['ID'].count())
education_split.rename(columns={'ID':'Number of customers'},inplace=True)
                      
ax = education_split[['Number of customers']].plot.bar(title='Clientes segmentados pelo nivel de educação', 
                                                      legend=True, table=False,
                                                      grid=False, subplots=False, 
                                                      figsize=(12,7), color='#EA1D2C',
                                                      fontsize = 15, stacked=False)

plt.ylabel('Number of customers\n', horizontalalignment='center', fontstyle='normal',
           fontsize='large', fontfamily='verdana')

plt.xlabel('\n Education',horizontalalignment='center',fontstyle='normal',
           fontsize='large',fontfamily='verdana')

plt.title('Customers by education level \n', horizontalalignment='center',fontstyle='normal',
          fontsize='22',fontfamily='verdana')

plt.legend(loc='upper right', fontsize='medium')
plt.xticks(rotation=0, horizontalalignment='center')
plt.yticks(rotation=0, horizontalalignment='right')

x_labels=np.array(education_split[['Number of customers']])

def add_value_labels(ax,spacing=5):
    for rect in ax.patches:
        y_value=rect.get_height()
        x_value= rect.get_x() + rect.get_width()/2
        space=spacing
        va='bottom'
        
        if y_value < 0:
            space *= -1
            va = 'top'
        label = '{:.0f}'.format(y.value)
        
plt.savefig('EducationLevel')

#splitting them by their relationship status
#there's also a pattern here, most of the customers have a partner
relationship_split=base_final[['ID','Marital_Status']]
sectors2 = relationship_split.groupby('Marital_Status')
relationship_split = pd.DataFrame(sectors2['ID'].count())
relationship_split.rename(columns={'ID':'Number of customers'},inplace=True)
                      
ax2 = relationship_split[['Number of customers']].plot.bar(title='Clientes segmentados pelo estado de relacionamento', 
                                                      legend=True, table=False,
                                                      grid=False, subplots=False, 
                                                      figsize=(12,7), color='#EA1D2C',
                                                      fontsize = 15, stacked=False)

plt.ylabel('Number of customers \n', horizontalalignment='center', fontstyle='normal',
           fontsize='large', fontfamily='verdana')

plt.xlabel('\n Relationship status',horizontalalignment='center',fontstyle='normal',
           fontsize='large',fontfamily='verdana')

plt.title('Customers by relationship status \n', horizontalalignment='center',fontstyle='normal',
          fontsize='22',fontfamily='verdana')

plt.legend(loc='upper right', fontsize='medium')
plt.xticks(rotation=0, horizontalalignment='center')
plt.yticks(rotation=0, horizontalalignment='right')

x_labels2=np.array(relationship_split[['Number of customers']])

def add_value_labels2(ax2,spacing=5):
    for rect in ax2.patches:
        y_value=rect.get_height()
        x_value= rect.get_x() + rect.get_width()/2
        space=spacing
        va='bottom'
        
        if y_value < 0:
            space *= -1
            va = 'top'
        label = '{:.0f}'.format(y.value)
        
plt.savefig('RelationshipStatus')

#Taking a look at the rest of the columns

#A good part of them Accepted Campaign 5 and Campaing 1 and 4
#Most part of them don't have Children
#Neither of these customers complained

binaries = ['Kidhome', 'Teenhome','AcceptedCmp1','AcceptedCmp2',
            'AcceptedCmp3','AcceptedCmp4','AcceptedCmp5',
            'Complain']

fig3, axes = plt.subplots(nrows=3, ncols=3,figsize=(15,12))

for i, item in enumerate(binaries):
    
    if i<3:
        ax=base_final[item].value_counts().plot(
            kind='bar',ax=axes[i,0],
            rot=0,color='#f3babc')
        ax.set_title(item)
            
    elif i>=3 and i<6:
            ax=base_final[item].value_counts().plot(
                kind='bar',ax=axes[i-3,1],
                rot=0, color='#9b9c9a')
            ax.set_title(item)
            
    elif i<9:
            ax=base_final[item].value_counts().plot(
                kind='bar',ax=axes[i-6,2],rot=0,
                color='#ec838a')
            ax.set_title(item)

#Setting a heatmap of the correlations between the main aspects of these customers
sns.set(style='white')
corr=base_final.drop(columns=['Complain','Z_CostContact','Z_Revenue']).corr()

mascara = np.zeros_like(corr, dtype=bool)
mascara[np.triu_indices_from(mascara)]=True

f, ax =plt.subplots(figsize=(18,15))
cmap = sns.diverging_palette(220,10, as_cmap=True)

sns.heatmap(corr, mask=mascara, cmap=cmap, vmax=.7, center=0,
           square=True, linewidth=.5, cbar_kws={'shrink':.5})

#Creating a simple array to plot the percentages of their purchases type

#It's possible to see that most of them buy on Stores or Catalog

TotalDealsPurchases=base_final['NumDealsPurchases'].sum()
TotalWebPurchases=base_final['NumWebPurchases'].sum()
TotalCatalogPurchases=base_final['NumCatalogPurchases'].sum()
TotalStorePurchases=base_final['NumStorePurchases'].sum()

TotalPurchases=TotalDealsPurchases+TotalWebPurchases+TotalCatalogPurchases+TotalStorePurchases

pctDeals=TotalDealsPurchases/TotalPurchases
pctWeb=TotalWebPurchases/TotalPurchases
pctCatalog=TotalCatalogPurchases/TotalPurchases
pctStore=TotalStorePurchases/TotalPurchases

ComportamentoCompras = [pctDeals,pctWeb,pctCatalog,pctStore]
ComportamentoCompras = pd.to_numeric(ComportamentoCompras,errors='coerce')
ComportamentoCompras = ComportamentoCompras.astype('float')

fig, ax = plt.subplots(figsize=(10,8))
pieLabels=['Deals','Web','Catalog','Store']
ax.pie(ComportamentoCompras, labels=pieLabels, autopct='%1.1f%%',
         startangle=90)

ax.axis('equal')
plt.tight_layout()
plt.title('High growth predicted users type of purchase')
plt.show()

#Creating a simple array to understand their amount of spent money

#Most of the amount spent was on Wines and another great part on Meat

TotalWine=base_final['MntWines'].sum()
TotalFruits=base_final['MntFruits'].sum()
TotalMeat=base_final['MntMeatProducts'].sum()
TotalFish=base_final['MntFishProducts'].sum()
TotalSweet=base_final['MntSweetProducts'].sum()
TotalGold=base_final['MntGoldProds'].sum()


TotalAmount=TotalWine+TotalFruits+TotalMeat+TotalFish+TotalSweet+TotalGold

pctWine=TotalWine/TotalAmount
pctFruits=TotalFruits/TotalAmount
pctMeat=TotalMeat/TotalAmount
pctFish=TotalFish/TotalAmount
pctSweet=TotalSweet/TotalAmount
pctGold=TotalGold/TotalAmount


ValorCompras = [pctWine,pctFruits,pctMeat,pctFish,pctSweet,pctGold]
ValorCompras = pd.to_numeric(ValorCompras,errors='coerce')
ValorCompras = ValorCompras.astype('float')

fig, ax = plt.subplots(figsize=(10,8))
pieLabels=['Wine','Fruits','Meat','Fish','Sweet','Gold']
ax.pie(ValorCompras, labels=pieLabels, autopct='%1.1f%%',
         startangle=90)

ax.axis('equal')
plt.tight_layout()
plt.title('High growth predicted users according to the spent amount')
plt.show()



