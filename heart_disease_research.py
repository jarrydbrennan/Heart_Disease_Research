# import libraries
import codecademylib3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency

# load data
heart = pd.read_csv('heart_disease.csv')

print(heart.head(10))

sns.boxplot(heart.heart_disease, heart.thalach)
plt.show()
plt.clf()

thalach_hd = heart.thalach[heart.heart_disease == 'presence']
thalach_no_hd = heart.thalach[heart.heart_disease == 'absence']

thalach_mean_difference = np.mean(thalach_hd) - np.mean(thalach_no_hd)
thalach_median_difference = thalach_hd.median() - thalach_no_hd.median()
print('The mean difference in thalach between patients with and without heart disease is: ', thalach_mean_difference)
print('The median difference in thalach between patients with and without heart disease is: ', thalach_median_difference)

tstat,pval = ttest_ind(thalach_hd, thalach_no_hd)
print('p-value from Two Sample T-Test:', pval)
print('With a p-value < 0.05, there is significant difference in the average thalach for people with heart disease compared to those without, thus we reject the null hypothesis.')

sns.boxplot(heart.heart_disease, heart.trestbps)
plt.show()
plt.clf()

trest_hd = heart.trestbps[heart.heart_disease == 'presence']
trest_no_hd = heart.trestbps[heart.heart_disease == 'absence']
tstat,pval = ttest_ind(trest_hd, trest_no_hd)
print(print('With a p-value of ',pval,'resting blood pressure is significant factor in the presence of heart disease'))

sns.boxplot(heart.heart_disease, heart.age)
plt.show()
plt.clf()

age_hd = heart.age[heart.heart_disease == 'presence']
age_no_hd = heart.age[heart.heart_disease == 'absence']
tstat,pval = ttest_ind(age_hd, age_no_hd)
print('With a p-value of ',pval,'age is a significant factor in the presence of heart disease')

sns.boxplot(heart.heart_disease, heart.chol)
plt.show()
plt.clf()

chol_hd = heart.chol[heart.heart_disease == 'presence']
chol_no_hd = heart.chol[heart.heart_disease == 'absence']
tstat,pval = ttest_ind(chol_hd, chol_no_hd)
print('With a p-value of ',pval,'cholesterol is not a significant factor in the presence of heart disease')

sns.boxplot(heart.cp, heart.thalach)
plt.show()
plt.clf()

thalach_typical = heart.thalach[heart.cp == 'typical angina']
thalach_asymptom = heart.thalach[heart.cp == 'asymptomatic']
thalach_nonangin = heart.thalach[heart.cp == 'non-anginal pain']
thalach_atypical = heart.thalach[heart.cp == 'atypical angina']

fstat,pval = f_oneway(thalach_typical,thalach_asymptom, thalach_nonangin, thalach_atypical)
print('With a p-value of',pval,'there is at least one pair of chest pain categories for which people in those categories have significantly different thalach.')

tukey_result = pairwise_tukeyhsd(heart.thalach, heart.cp, 0.05)
print(tukey_result)

Xtab = pd.crosstab(heart.cp, heart.heart_disease)
print(Xtab)

chi2,pval,dof,expected = chi2_contingency(Xtab)
print('With a p-value of',pval,'there is a significant association between chest pain type and whether or not someone is diagnosed with heart disease.')

Gtab = pd.crosstab(heart.sex, heart.heart_disease)
chi2,pval,dof,expected = chi2_contingency(Gtab)
print(pval)
print('With a p-value of',pval,'there is a significant difference between males and females and heart disease.')

sns.boxplot(heart.sex, heart.chol)
plt.show()
plt.clf()

female_chol = heart.chol[heart.sex == 'female']
male_chol = heart.chol[heart.sex == 'male']
tstat,pval = ttest_ind(female_chol, male_chol)
print('With a p-value of ',pval,'there is a significant difference in cholesterol between males and females.')