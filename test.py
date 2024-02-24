import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import model

time_use=[]
EPOCH =100
sele = [4, 5]
x = pd.read_csv('data/SEEDS_train.csv')
y = pd.read_csv('data/SEEDS_test.csv')
data = np.concatenate((x, y), axis=0)

X = data[:, 1:]
y = data[:, 0]

le = LabelEncoder()
y = le.fit_transform(y)

X = X.astype(np.float32)
y = y.astype(np.float32)

X_train1, x_test, y_train1, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train1, y_train1, test_size=0.3)
num = X_train.shape[0]
b_s = 12
fea_num = len(X.T)
class_num = int(max(y) + 1)

Self_CovTabnet_model = model.Self_CovTabNetClassifier()
Self_CovTabnet_model.fit(
        X_train, Y_train,
        eval_set=[(X_val, Y_val)],
        max_epochs=EPOCH,
        patience=20,
        batch_size=b_s,
        virtual_batch_size=b_s,
        hy_num=sele[0],
        layer_num=sele[1]
    )
y_pred = Self_CovTabnet_model.predict(x_test)
AC = accuracy_score(y_test, y_pred)
PE = precision_score(y_test, y_pred, average='macro')
RE = recall_score(y_test, y_pred, average='macro')
F1 = f1_score(y_test, y_pred, average='macro')
Indexes = np.array([AC, PE, RE, F1])

print("Self-CovTabNet Accuracy: {:.8f}".format(AC))

