#!/usr/bin/env python
import argparse
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
from cycler import cycler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--project')
args, _ = parser.parse_known_args()

model_dir = os.environ['AIP_MODEL_DIR']

np.random.seed(314159)
train_txn = pd.read_csv(f'/gcs/{args.project}/data/train_transaction.csv')
test_txn = pd.read_csv(f'/gcs/{args.project}/data/test_transaction.csv')
train_id = pd.read_csv(f'/gcs/{args.project}/data/train_identity.csv')


set(train_txn.columns) - set(test_txn.columns)

isFraud = train_txn[train_txn['isFraud']==1]['isFraud']
isNotFraud = train_txn[train_txn['isFraud']==0]['isFraud']
print('Num fraud: {}\nNon non-fraud: {}\nPercent fraud: {}'.format(isFraud.count(), isNotFraud.count(), isFraud.count()/(isNotFraud.count()+isFraud.count())))

#drop categorical features with high cardinality
new_df = train_txn.drop(columns=['P_emaildomain', 'R_emaildomain'])

#one-hot remaining categorical features
new_df = pd.get_dummies(new_df, columns=['ProductCD'], prefix=['ProductCD'])
new_df = pd.get_dummies(new_df, columns=['card4'], prefix=['card4'])
new_df = pd.get_dummies(new_df, columns=['card6'], prefix=['card6'])



#drop columns that have >=25% missing values
size = train_txn.shape[0]
#new_df = new_df.dropna(axis=1, thresh=(.25 * size))

#drop rows that still have missing values (won't drop more than 25% of dataset, guaranteed above)
#new_df = new_df.dropna(axis=0)

#binary encode M1-9
encode = lambda truth: 1 if truth=="T" else 0
for i in range(1,10):
  label = "M" + str(i)
  new_df[label] = new_df[label].apply(encode)


lim_corr = new_df.corrwith(new_df['isFraud'])

lim_corr.filter(regex='[^V\d+]', axis=0).sort_values(ascending=False).head(50)

lim_corr.filter(regex='[^V\d+]', axis=0).sort_values().head(50)

frauds = train_txn.loc[train_txn['isFraud'] == 1]
frauds

notfraud = train_txn.loc[train_txn['isFraud'] == 0].sample(n=20663)
even = pd.concat([frauds,notfraud], ignore_index=True)
even = even.loc[:,~even.columns.str.startswith('V')]

trimmed = train_txn.loc[:,~train_txn.columns.str.startswith('V')]

total = len(trimmed)

trimmed[["card3","isFraud"]][trimmed["card3"]==125]

total = len(even)

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

_KEEP_COLUMNS_MODEL_1 = ['TransactionAmt']

_KEEP_COLUMNS_MODEL_1_5 = ['ProductCD', 'TransactionAmt', 'card1', 'card2', 'card3', 'card4', 
                      'card5', 'card6', 'P_emaildomain', 'isFraud']

_KEEP_COLUMNS_MODEL_2 = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 
                      'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain']

_RESPONSE = 'isFraud'

_KEEP_COLUMNS_MODEL_TxnPcd = ['TransactionAmt', 'ProductCD']


train_txn_copy = train_txn.copy()
y_df = train_txn_copy[_RESPONSE]
x_df = train_txn_copy
# remove the target label from our training dataframe...
del x_df[_RESPONSE]

# stratify on the target column to keep the approximate balance of positive examples since it's so imbalanced
x_train_df, x_test_df, y_train_df, y_test_df = \
  train_test_split(x_df, y_df, test_size=0.25, stratify=y_df)
x_train_norm_df = x_train_df[_KEEP_COLUMNS_MODEL_TxnPcd]
x_train_norm_df.TransactionAmt = (x_train_norm_df.TransactionAmt - x_train_df.TransactionAmt.mean()) / x_train_df.TransactionAmt.std()
x_train_oh_df = pd.get_dummies(x_train_norm_df)

inputs = Input(shape=(x_train_oh_df.values.shape[1],))
preds = Dense(1, activation='sigmoid')(inputs)
model = Model(inputs=inputs, outputs=preds)
model.compile(optimizer=SGD(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train_oh_df.values, y_train_df.values, batch_size=512, epochs=10, shuffle=False)


model.save(model_dir)