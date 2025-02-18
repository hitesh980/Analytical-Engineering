#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


df = pd.read_csv(r"/Users/allakkihome/Downloads/archive (7)/Uncleaned_DS_jobs.csv")


# In[30]:


df.info()


# In[ ]:





# In[32]:


df.head(4)


# In[33]:


print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")


# In[34]:


df = df[df['Salary Estimate'] != '-1']


# In[35]:


print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")


# In[36]:


salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
salary = df['Salary Estimate'].apply(lambda x: x.replace("k"," ").replace("$",""))


# In[37]:


print(df['Salary Estimate'])


# In[38]:


print(salary)


# In[39]:


salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])


# In[ ]:





# In[40]:


salary = df['Salary Estimate'].apply(lambda x: x.replace("k"," ").replace("$",""))


# In[41]:


print(salary)


# In[42]:


df['Salary Estimate'] = df['Salary Estimate'].apply(lambda x: x.split('(')[0] if isinstance(x, str) else x)


# In[43]:


print(df['Salary Estimate'])


# In[44]:


df['Salary Estimate'] = df['Salary Estimate'].apply(lambda x: x.replace("K"," ").replace("$",""))


# In[45]:


print(df['Salary Estimate'])


# In[46]:


df['hourly'] = df['Salary Estimate'].astype(str).apply(lambda x: 1 if 'per hour' in x.lower() else 0)


# In[47]:


df.head(100)


# In[ ]:





# In[48]:


df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary: ' in x.lower() else 0)


# In[49]:


df.head(5)


# In[50]:


df['min_salary'] = df['Salary Estimate'].apply(lambda x: x.split('-')[0] )


# In[ ]:





# In[51]:


df['max_salary'] = df['Salary Estimate'].apply(lambda x: x.split('-')[1] )


# In[ ]:





# In[ ]:





# In[52]:


df['max_salary'] = df['Salary Estimate'].apply(lambda x: int(x.split('-')[1].strip()))
df['min_salary'] = df['Salary Estimate'].apply(lambda x: int(x.split('-')[0].strip()))


# In[53]:


df['avg_salary'] = (df.min_salary+df.max_salary)/2


# In[54]:


df.head(56)


# In[55]:


df['Company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating']<0 else x['Company Name'][:-4],axis =1)


# In[56]:


df.head(5)


# In[92]:


df['job_state'] = df['Location'].apply(lambda x: x.split(',')[-1])


# In[93]:


df.head(7)


# In[59]:


df.job_state.value_counts()


# In[60]:


df['same_state'] = df.apply(lambda x : 1 if x.Location == x.Headquarters else 0,axis =1)


# In[61]:


#age of the company
from datetime import date
df['age'] = df['Founded'].apply(lambda x: x if x <0 else date.today().year -x)


# In[62]:


df.head(6)


# In[63]:


#parsing of job description (python etc..)
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)


# In[64]:


df['python_yn'].value_counts()


# In[65]:


df['r_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)


# In[66]:


df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)


# In[67]:


df['spark_yn'].value_counts()


# In[68]:


df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)


# In[69]:


df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)


# In[70]:


#EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[71]:


df.head()


# In[78]:



df.head()


# In[77]:


df.columns


# In[76]:


df.drop(columns=['index'], inplace=True, errors='ignore')


# In[79]:


def title_simplifier(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'na'
    
def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'sr' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
            return 'senior'
    elif 'jr' in title.lower() or 'jr.' in title.lower():
        return 'jr'
    else:
        return 'na'
		
## Job title and seniority 
		


# In[82]:


df['job_simp'] = df['Job Title'].apply(title_simplifier)


# In[154]:


df.job_simp.value_counts()


# In[85]:


df['seniority'] = df['Job Title'].apply(seniority)
df.seniority.value_counts()


# In[88]:


df.job_state.value_counts()


# In[94]:


pd.set_option('display.max_rows', None)  # Show all rows
print(df['job_state'].value_counts())


# In[99]:


df['job_state'] = df['job_state'].apply(lambda x: 'NJ' if x == 'New Jersey' else ('TX' if x == 'Texas' else x))
df.job_state.value_counts()


# In[98]:


df = df[~df['job_state'].isin(['California'])]


# In[100]:


#job desc lenght
df['desc_len'] = df['Job Description'].apply(lambda x: len(x))


# In[101]:


df['desc_len']


# In[105]:


#competitor Count
df['Num_Comp'] = df['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)


# In[109]:


print(df[['Num_Comp', 'Competitors']])


# In[116]:


df.describe()


# In[117]:


df.Rating.hist()


# In[119]:


df.avg_salary.hist()


# In[120]:


df.age.hist()


# In[121]:


df.desc_len.hist()


# In[123]:


df.boxplot(column = ['age','avg_salary','Rating'])


# In[124]:


df.boxplot(column = ['Rating'])


# In[125]:


df[['age','avg_salary','Rating','desc_len']].corr()


# In[128]:


cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(df[['age','avg_salary','Rating','desc_len','Num_Comp']].corr(),vmax=.3, center=0, cmap=cmap,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[130]:


df_cat = df[['Location', 'Headquarters', 'Size','Type of ownership', 'Industry', 'Sector', 'Revenue', 'Company_txt', 'job_state','same_state', 'python_yn', 'r_yn',
       'spark_yn', 'aws', 'excel', 'job_simp', 'seniority']]


# In[140]:


for i in df_cat.columns:
    cat_num = df_cat[i].value_counts()
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x = cat_num.index,y= cat_num)
    chart.set_xticklabels(chart.get_xticklabels(),rotation = 90)
    plt.show()


# In[143]:


for i in df_cat[['Location','Headquarters','Company_txt']].columns:
    cat_num = df_cat[i].value_counts()[:20]
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x = cat_num.index,y= cat_num)
    chart.set_xticklabels(chart.get_xticklabels(),rotation = 90)
    plt.show()


# In[144]:


df.columns


# In[145]:


pd.pivot_table(df, index = 'job_simp',values = 'avg_salary')


# In[146]:


pd.pivot_table(df, index = ['job_simp','seniority'] ,values = 'avg_salary')


# In[147]:


pd.pivot_table(df,index = 'job_state', values ='avg_salary').sort_values('avg_salary',ascending = False)


# In[149]:


pd.pivot_table(df,index = ['job_state','job_simp'], values ='avg_salary',aggfunc = 'count').sort_values('job_state',ascending = False)


# In[156]:


pd.pivot_table(df[df.job_simp =='data scientist'] ,index = ['job_state','job_simp'], values ='avg_salary').sort_values('avg_salary',ascending = False)


# In[157]:


df.columns


# In[167]:


df_pivots = df[['Rating','Industry','Sector','Revenue','Num_Comp','hourly','employer_provided','python_yn','r_yn','spark_yn','aws','excel','desc_len','Type of ownership','avg_salary']]


# In[170]:


for i in df_pivots.columns:
    print(i)
    print(pd.pivot_table(df_pivots, index=i, values='avg_salary').sort_values('avg_salary', ascending = False))


# In[174]:


pd.pivot_table(df_pivots,index ='Revenue',columns = 'python_yn', values = 'avg_salary',aggfunc = 'count')


# In[180]:



get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[183]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')


# In[184]:


words = " ".join(df['Job Description'])

def punctuation_stop(text):
    """Remove punctuation and stop words"""
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered

words_filtered = punctuation_stop(words)
text = " ".join(words_filtered)

wc = WordCloud(background_color="white", random_state=1, stopwords=STOPWORDS, max_words=2000, width=800, height=1500)
wc.generate(text)

plt.figure(figsize=[10,10])
plt.imshow(wc, interpolation="bilinear")  
plt.axis('off')
plt.show()


# In[205]:


#model building
#choosing relevant columns
import numpy as np
df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','Num_Comp','hourly','employer_provided',
             'job_state','same_state','age','python_yn','spark_yn','aws','excel','job_simp','seniority','desc_len']]


# In[192]:


df_dum = pd.get_dummies(df_model)
df_dum


# In[193]:


#train test split
from sklearn.model_selection import train_test_split


# In[198]:


X = df_dum.drop('avg_salary', axis =1)
y = df_dum.avg_salary.values


# In[199]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state =42)


# In[201]:


#linear regression
import statsmodels.api as sm
 
X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()


# In[202]:


from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score


# In[203]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[206]:


np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))


# In[207]:


# lasso regression 
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]


# In[212]:


#random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np

rf = RandomForestRegressor(criterion='absolute_error')  # Explicitly set criterion
neg_mae = np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))
print("Negative MAE:", neg_mae)

parameters = {
    'n_estimators': range(10, 300, 10),
    'criterion': ('absolute_error',),  # Only use 'absolute_error' here
    'max_features': ('auto', 'sqrt', 'log2')
}

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)

print("Best Score:", gs.best_score_)
print("Best Estimator:", gs.best_estimator_)


# In[213]:


# test ensemble
sklearn.metrics import mean_absolute_error
import pickle

tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

print(mean_absolute_error(y_test, tpred_lm))
print(mean_absolute_error(y_test, tpred_lml))
print(mean_absolute_error(y_test, tpred_rf))

print(mean_absolute_error(y_test, (tpred_lm + tpred_rf) / 2))

pickl = {'model': gs.best_estimator_}
pickle.dump(pickl, open('model_file' + ".p", "wb"))

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

print(model.predict(np.array(list(X_test.iloc[1, :])).reshape(1, -1))[0])
print(list(X_test.iloc[1, :]))


# In[ ]:




