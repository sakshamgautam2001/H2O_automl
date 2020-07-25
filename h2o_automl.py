import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_rows = 999

import h2o
from h2o.automl import H2OAutoML

dataset = pd.ExcelFile('data/bank_term_deposit_marketing_analysis.xlsx')

client = pd.read_excel(dataset, 'CLIENT_INFO')
loan = pd.read_excel(dataset, 'LOAN_HISTORY')
market = pd.read_excel(dataset, 'MARKETING HISTORY')
subscription = pd.read_excel(dataset, 'SUBSCRIPTION HISTORY')
#client.head()
#subscription.head()

#############Building overall data frame#############

dataframe = pd.merge(client, loan, on=['ID'])
dataframe = pd.merge(dataframe, market, on=['ID'])
dataframe = pd.merge(dataframe, subscription, on=['ID'])
#dataframe.head()
#dataframe.count()
dataframe = dataframe.drop(['ID'], axis=1)
#dataframe.count()


h2o.init()

H2O = h2o.H2OFrame(dataframe)

train, test = H2O.split_frame(ratios=[.75])
X = train.columns
a = "TERM_DEPOSIT"
X.remove(a)

auto_ml = H2OAutoML(max_runtime_secs=400,
                seed=1,
                balance_classes=False,
                project_name='Completed'
                )

leadboard = auto_ml.leaderboard
leadboard.head(rows=leadboard.nrows)  

stack_ens = auto_ml.leader

metal = h2o.get_model(stack_ens.metal()['name'])       #metal => metalearner

metal.varimp()

final_model = h2o.get_model('XGBoost_grid__1_AutoML_20200608_075205_model_2')      #XG-Boost is best model
final_model.model_performance(test)








