"""Experiment 5: AdaBoost (decision stumps) vs random forest, Adult style data."""
import math,random
from experiments_common import make_adult,split_indices,tree_fit,tree_predict,forest_fit,forest_predict,metrics
X,y=make_adult(); tr,te=split_indices(500,seed=42); n=len(tr); weights=[1/n]*n; models=[]
for _ in range(30):
 xx=[X[i] for i in tr]; yy=[y[i] for i in tr]; t=tree_fit(xx,yy,depth=1,leaf=5,weights=weights); pred=[tree_predict(t,row) for row in xx]; err=sum(weights[i]*(pred[i]!=yy[i]) for i in range(n)); err=min(.499,max(.001,err)); alpha=.5*math.log((1-err)/err); weights=[weights[i]*math.exp(-alpha*(1 if pred[i]==yy[i] else -1)) for i in range(n)]; total=sum(weights); weights=[v/total for v in weights]; models.append((t,alpha))
ada=[]
for i in te:
 score=sum(a*(1 if tree_predict(t,X[i]) else -1) for t,a in models); ada.append(int(score>=0))
rf=forest_fit([X[i] for i in tr],[y[i] for i in tr],count=31); rp=[forest_predict(rf,X[i]) for i in te]; actual=[y[i] for i in te]
print('Adult-style synthetic data:',500,'rows; 400 train / 100 test'); print('AdaBoost (30 weighted decision stumps):',metrics(actual,ada)); print('Random Forest (31 trees):',metrics(actual,rp))
