#!/usr/bin/env python
# coding: utf-8

# In[39]:


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
from efficient_apriori import apriori
import warnings
warnings.filterwarnings('ignore')


# In[88]:


data = pd.read_csv('records-for-2011.csv', index_col=0)
print('属性类别数:', len(data.columns))
print('总行数:', len(data))


# In[89]:


#数据示例
data.head(5)


# In[90]:


#每一列的属性和名称
num_fields = data.select_dtypes(include=np.number).columns.values
nom_fields = data.select_dtypes(exclude=np.number).columns.values
all_fields=[]
for field in range(len(num_fields)):
    all_fields.append(num_fields[field])
for field in range(len(nom_fields)):
    all_fields.append(nom_fields[field])
print("属性:", all_fields)

print('标称属性:', nom_fields)

print('数值属性:', num_fields)  


# In[91]:


#每个属性的空值统计
for field in all_fields:
    print(field,":",data[field].isnull().sum())


# In[92]:


#分析犯罪地点和犯罪类型两个属性
print('数据行数:', len(data))
data = data[['Location', 'Incident Type Id']]
data = data.dropna(how='any')
print('缺失部分剔除后数据行数:', len(data))
#data = data.drop('Agency',axis=1, inplace=True)


# In[96]:


data.head()


# In[107]:


apriori_data = []
for s in data.iterrows():
    #print(s[1][1])
    apriori_data.append((s[1][0], s[1][1]))


# In[113]:


#利用aprori算法进行频繁算法
itemsets, rules = apriori(apriori_data, min_support=0.00005,  min_confidence=0.3)


# In[114]:


itemsets


# In[115]:


#导出关联规则，计算其支持度和置信度
rules


# In[116]:


#计算关联规则与置信度
for rule in sorted(rules, key=lambda rule: rule.confidence):
  print(repr(rule), 'support:', rule.support, 'confidence:', rule.confidence,"\n")


# In[117]:


#使用Lift、卡方对规则进行评价
for rule in sorted(rules, key=lambda rule: rule.confidence):
  print(repr(rule), 'lift:', rule.lift, 'conviction:', rule.conviction,"\n")


# In[118]:


#规则可视化

def plot_bar(rules, data, title):
    plt.title(title)
    plt.xticks(range(len(data)),rules,rotation=90)
    plt.bar(range(len(data)), data)
    plt.show()

def visualization(big_rule_list):
    rules = []
    conf = []
    support = []
    lift = []
    for rule in big_rule_list:
        rules.append(repr(rule))
        conf.append(rule.confidence)
        support.append(rule.support)
        lift.append(rule.lift)
    plot_bar(rules, support, 'rule-support figure')
    plot_bar(rules, conf, 'rule-confidence figure')
    plot_bar(rules, lift, 'rule-lift figure')

visualization(sorted(rules, key=lambda rule: rule.confidence)[:20])


# In[ ]:




