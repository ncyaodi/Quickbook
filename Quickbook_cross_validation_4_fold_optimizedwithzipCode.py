#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas
import mba263
import numpy as np



import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold 
# import uszipcode
pandas.options.mode.chained_assignment = None




data_with_dummies=pandas.read_csv('data_with_dummies_byLargeState_withIncome.csv')

print(data_with_dummies)


# In[9]:


# varlist=['bizflag','numords','dollars','last','sincepurch','version1','owntaxprod','upgraded','sex_F','sex_M','sex_U', 
#          'median_house_income',
#         'zip_bins_AL','zip_bins_AR','zip_bins_AZ','zip_bins_CA','zip_bins_CO','zip_bins_CT','zip_bins_DC',
#         'zip_bins_DE','zip_bins_FL','zip_bins_GA','zip_bins_HI','zip_bins_IA','zip_bins_ID',
#         'zip_bins_IL','zip_bins_IN','zip_bins_INTL','zip_bins_KS','zip_bins_KY','zip_bins_LA',
#         'zip_bins_MA','zip_bins_MD','zip_bins_ME','zip_bins_MI','zip_bins_MN','zip_bins_MO','zip_bins_MS',
#         'zip_bins_MS','zip_bins_MT','zip_bins_NA','zip_bins_NC','zip_bins_ND','zip_bins_NE',
#         'zip_bins_NH','zip_bins_NJ','zip_bins_NM','zip_bins_NV','zip_bins_NY','zip_bins_OH','zip_bins_OK',
#          'zip_bins_OR','zip_bins_PA','zip_bins_PR','zip_bins_RI','zip_bins_SC','zip_bins_SD','zip_bins_TN',
#         'zip_bins_TX','zip_bins_UT','zip_bins_VA','zip_bins_VT','zip_bins_WA','zip_bins_WI',
#         'zip_bins_WV','zip_bins_WY']



varlist=['bizflag','numords','dollars','last','sincepurch','version1','owntaxprod','upgraded','sex_F','sex_M','sex_U', 
         'median_house_income',
         'zip_bins_2','zip_bins_3','zip_bins_4','zip_bins_5','zip_bins_6','zip_bins_7','zip_bins_8',
         'zip_bins_9','zip_bins_10','zip_bins_11','zip_bins_12','zip_bins_13','zip_bins_14']


# In[ ]:





# In[10]:


####### Split data into training set and test set #######


# In[23]:




# In[22]:


y_dummy = data_with_dummies['res1'].to_frame()
X_dummy = data_with_dummies[varlist]
#mba263.tabulate(X_dummy['zip'])

#data_bin1=data_with_dummies[data_with_dummies['zip_bins']==1]

#mba263.tabulate(data_bin1['zip'])


# In[12]:


n_fold = 3
kf = KFold(n_splits=n_fold, random_state=1, shuffle=True)
kf.get_n_splits(X_dummy)


# In[ ]:





# In[13]:


i = 1

for train_index, test_index in kf.split(X_dummy):

    globals()['X_train_' + str(i)]=X_dummy.loc[train_index]
    globals()['X_test_' + str(i)]=X_dummy.loc[test_index]
    globals()['y_train_' + str(i)]=y_dummy.loc[train_index]
    globals()['y_test_' + str(i)]=y_dummy.loc[test_index]
    i=i+1
    
print(y_test_1)


# In[ ]:





# In[ ]:





# Logistic Regression with no interactions

# In[ ]:





# In[ ]:





# In[14]:


##### Train Models ######


# In[15]:


## Logistic Regression


# In[16]:


for i in range(1,n_fold+1):
    globals()['model_logit_' + str(i)] = mba263.logit_reg(globals()['y_train_' + str(i)]['res1'],                                                      globals()['X_train_' + str(i)])
    
    print('Completed %s iteration of fold modeling' % i)


# In[ ]:





# In[17]:


for i in range(1,n_fold+1):
    globals()['gain_logit_' + str(i)]    =mba263.gain(globals()['y_test_' + str(i)]['res1'],                 globals()['model_logit_' + str(i)].predict(globals()['X_test_' + str(i)]),                 bins =100)
    
    plt.plot(globals()['gain_logit_' + str(i)],label='logit model %s' % i)
    
plt.plot([0,100],[0,1])
plt.legend()


# In[ ]:
results = mba263.odds_ratios(model_logit_1)

# print(mba263.odds_ratios(model_rf1_1))

# Logistic Model 1 slightly better
# In[18]:



#### Neural Network Modeling 


# In[19]:


for i in range(1,n_fold+1):
    globals()['model_nn_' + str(i)] = mba263.neural_network(globals()['y_train_' + str(i)]['res1'],                                                      globals()['X_train_' + str(i)])
    
    print('Completed %s iteration of fold modeling' % i)


# In[20]:


for i in range(1,n_fold+1):
    globals()['Result_test_' + str(i)] = globals()['y_test_' + str(i)]
    
    globals()['Result_test_' + str(i)]['p_nn'] =    globals()['model_nn_' + str(i)].predict(globals()['X_test_' + str(i)])
    
    globals()['gain_nn_' + str(i)]    =mba263.gain(globals()['y_test_' + str(i)]['res1'],                 
                                                   globals()['Result_test_' + str(i)]['p_nn'],                 
                                                   bins =100)
    
    plt.plot(globals()['gain_nn_' + str(i)],label='neural network model %s' % i)
    
plt.plot([0,100],[0,1])
plt.legend()


# In[21]:


# NN Model 1 has the best performance


# In[ ]:





# In[22]:


#### Random Forest Model. #####


# In[23]:


for i in range(1,n_fold+1):
    globals()['model_rf1_' + str(i)] = mba263.random_forest(globals()['y_train_' + str(i)]['res1'],                                                            
                                                            globals()['X_train_' + str(i)], 
                                                            trees=1000,leaf_nodes=500)
    
    print('Completed %s iteration of fold modeling' % i)


# In[24]:


for i in range(1,n_fold+1):
    
    
    globals()['Result_test_' + str(i)]['p_rf1'] =    globals()['model_rf1_' + str(i)].predict(globals()['X_test_' + str(i)])
    
    globals()['gain_rf1_' + str(i)]    =mba263.gain(globals()['y_test_' + str(i)]['res1'],                 globals()['Result_test_' + str(i)]['p_rf1'],                 bins =100)
    
    plt.plot(globals()['gain_rf1_' + str(i)],label='Random Forest #1 model %s' % i)
    
plt.plot([0,100],[0,1])
plt.legend()


# In[25]:


# No obvious difference. Choose model 1.


# In[ ]:





# In[26]:


### Compare models

# Pick the best model within each model
# In[37]:


plt.plot(gain_logit_1,label='Best logit model')
plt.plot(gain_nn_1,label='Best neural network model')
plt.plot(gain_rf1_1,label='Best random forest model')
plt.plot([0,100],[0,1])
plt.legend()


# In[ ]:





# In[28]:


####### Choose mailing list for wave 2. ######


# In[29]:


# Predict using the final model


# In[38]:


p_final = model_logit_1.predict(data_with_dummies[varlist])
data_with_dummies['p_final']=p_final


# In[ ]:





# In[31]:


# Sort out customers who didn't response in wave 1


# In[32]:


data_responsed=data_with_dummies[data_with_dummies['res1']==1]
data_non_response=data_with_dummies[data_with_dummies['res1']==0]


# In[ ]:





# In[33]:


# Assuming 50% dropout rate, choose customers based on break-even p-value 2.35%


# In[39]:


data_mail_wave2=data_non_response[data_non_response['p_final']>(0.0235*2)]
print(data_mail_wave2)


# In[ ]:


## Need to send 5622 mails in wave 2 with 3-fold


# In[ ]:


## Output maillist


# In[40]:


ArithmeticError(args)data_mail_wave2.to_csv('Mail_list_wave_2_4foldstudy_byRegion_withIncome.csv')


# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




