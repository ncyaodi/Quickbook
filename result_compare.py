#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:19:14 2020

@author: bxg_ellie
"""

#%%
import pandas
import mba263
import numpy as np


#%%

data_basic_3fold = pandas.read_csv('Mail_list_wave_2_3foldstudy.csv')
data_basic_4fold = pandas.read_csv('Mail_list_wave_2_4foldstudy.csv')
data_bystate_4fold = pandas.read_csv('Mail_list_wave_2_4foldstudy_byState.csv')
data_byregion_4fold = pandas.read_csv('Mail_list_wave_2_4foldstudy_byRegion_withIncome.csv')


mail1 = data_basic_3fold['id']
mail2 = data_basic_4fold['id']
mail3 = data_bystate_4fold['id']
mail4 = data_byregion_4fold['id']


#%%

mail_common_1 = [value for value in mail4 if value in mail3] 

mail_common_2 = [value for value in mail_common_1 if value in mail2] 

mail_common_3 = [value for value in mail_common_2 if value in mail1] 



print(len(mail_common_3))


gold_list = data_basic_3fold.loc[data_basic_3fold['id'].isin(mail_common_3)]

print(gold_list)

#%%

gold_list.to_csv('Mail_list_wave_2_golden_ID_list.csv')