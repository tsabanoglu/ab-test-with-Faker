#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 07:31:32 2024

@author: tugbasabanoglu
"""

import pandas as pd
import numpy as np
from faker import Faker
from scipy import stats

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# initialize a Faker object
fake = Faker()


### null-hypothesis: there is no significant change in user ratings after the app receives an update ###

# set up the datasets 
n_users = 1000 # each dataset with 1000 obs


# generate data for control group - let's call it group A
user_data_A = {
    'user_id': [fake.unique.uuid4() for _ in range(n_users)],
    'name': [fake.name() for _ in range(n_users)],
    'rating': np.random.normal(3.5, 0.8, n_users),
    'date': [fake.date_between(start_date='-2y', end_date='-1y') for _ in range(n_users)],
    'group': 'A'
    }

# generate data for test group or group B
user_data_B = {
    'user_id': [fake.unique.uuid4() for _ in range(n_users)],
    'name': [fake.name() for _ in range(n_users)],
    'rating': np.random.normal(4.0, 0.7, n_users),
    'date': [fake.date_between(start_date='-1y', end_date='today') for _ in range(n_users)],
    'group': 'B'
    }


# combine both sets into a single df
df_A = pd.DataFrame(user_data_A)
df_B = pd.DataFrame(user_data_B)
df = pd.concat([df_A, df_B], ignore_index=True)

# calculate summary statistics
mean_A = df_A['rating'].mean()
std_A = df_A['rating'].std()
mean_B = df_B['rating'].mean()
std_B = df_B['rating'].std()

# visualize the data to see the distribution

plt.figure(figsize=(12, 6))

# histogram
plt.subplot(1, 3, 1)
sns.histplot(df_A['rating'], color='blue', label='Group A', kde=False, bins=20)
sns.histplot(df_B['rating'], color='red', label='Group B', kde=False, bins=20)
plt.title('Histogram of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.legend()


### alternatively, if we have no knowledge about data's distribution and want to check for normality ###
# # Q-Q plot for group A
# plt.subplot(1, 2, 1)
# stats.probplot(df_A['rating'], dist="norm", plot=plt)
# plt.title('Q-Q Plot (Group A)')

# # Q-Q plot for group B
# plt.subplot(1, 2, 2)
# stats.probplot(df_B['rating'], dist="norm", plot=plt)
# plt.title('Q-Q Plot (Group B)')

# plt.tight_layout()
# plt.show()

### since data is normally distributed ###

# perform (independent) t-test on both groups
t_stat, p_value = stats.ttest_ind(df_A['rating'], df_B['rating'])

print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

# we will assume a significance level (alpha) of 0.05 
if p_value < 0.05:
    print("There is a statistically significant difference between the two groups.")
else:
    print("There is no statistically significant difference between the two groups.")
    
### decision: we will reject the null-hypothesis ###
    