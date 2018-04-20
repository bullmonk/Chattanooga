
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

shap.initjs()


# In[2]:


var_names=['u10']


# In[3]:


pm25_matrix_83 = pd.read_csv('', names=var_names, header=None)
pm25_19_response = pd.read_csv('', header=None)
pm25_matrix_83.shape, pm25_19_response.shape


# In[4]:


X_var = pm25_matrix_83[['']]
X_var.head()


# In[5]:


X = X_var[:12564]
X.shape


# In[6]:


y = pm25_19_response
len(y)


# In[7]:


X.columns = ['']


# In[8]:


from sklearn.model_selection import train_test_split
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=10)
dt = xgb.DMatrix(Xt.as_matrix(), label=yt.as_matrix())
dv = xgb.DMatrix(Xv.as_matrix(), label=yv.as_matrix())


# In[9]:


## tune best max_depth, min_child_weight
params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:linear'
}


gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(6,14)
    for min_child_weight in range(1,6)
]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))

    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain=dt,
        num_boost_round=300,
        seed=2,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
        
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
# Best (11,4)


# In[10]:


## tune best subsample and colsample
params['max_depth'] = 13
params['min_child_weight'] = 1

gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]

min_mae = float("Inf")
best_params = None

# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))

    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain = dt,
        num_boost_round=300,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )

    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))


# In[11]:


# tune best learning rate
params['subsample'] = 1
params['colsample_bytree'] = 0.8

min_mae = float("Inf")
best_params = None

for eta in [.3, .2, .1, .05, .01]:
    print("CV with eta={}".format(eta))

    # We update our parameters
    params['eta'] = eta

    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain = dt,
            num_boost_round=300,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10)

    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta

print("Best params: {}, MAE: {}".format(best_params, min_mae))


# In[12]:


## implement params
params = {'colsample_bytree': 0.8,
          'eta': 0.05,
          'eval_metric': 'mae',
          'max_depth': 13,
          'min_child_weight': 1,
          'objective': 'reg:linear',
          'subsample': 1}

model = xgb.train(
    params,
    dtrain=dt,
    num_boost_round=500,
    evals=[(dv, "Valid")],
    early_stopping_rounds=100,
    verbose_eval=25
)


# In[13]:


shap_values = model.predict(dv, pred_contribs=True)
shap.summary_plot(shap_values, Xv)


# In[15]:


## multiple linear regression
import statsmodels.api as sm
from scipy.stats.mstats import zscore

