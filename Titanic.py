"""Experiment 2: Titanic style logistic regression; 500 synthetic rows."""
import math
from experiments_common import make_titanic,split_indices,sigmoid,metrics
X,y=make_titanic(); tr,te=split_indices(500,seed=42); mu=[sum(X[i][j] for i in tr)/len(tr) for j in range(7)]; sd=[math.sqrt(sum((X[i][j]-mu[j])**2 for i in tr)/len(tr)) or 1 for j in range(7)]; Z=[[(r[j]-mu[j])/sd[j] for j in range(7)] for r in X]; w=[0.0]*8; rate=.08
for epoch in range(1800):
 grad=[0.0]*8
 for i in tr:
  z=[1]+Z[i]; e=sigmoid(sum(a*b for a,b in zip(w,z)))-y[i]
  for j in range(8):grad[j]+=e*z[j]/len(tr)
 w=[a-rate*b for a,b in zip(w,grad)]
p=[int(sigmoid(sum(a*b for a,b in zip(w,[1]+Z[i])))>=.5) for i in te]
print('Titanic synthetic data:',500,'rows; features pclass, sex, age, sibsp, parch, fare, embarked'); print('sex: 0=male, 1=female; embarked: 0=C, 1=Q, 2=S'); print('Logistic regression metrics:',metrics([y[i] for i in te],p)); print('Coefficients (intercept then standardized features):',[round(v,3) for v in w])
