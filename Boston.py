"""Experiment 1: Boston Housing style multiple linear regression; 500 synthetic rows."""
import random, math
from experiments_common import split_indices
random.seed(1); names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']; X=[]; y=[]
for _ in range(500):
 x=[random.random()*v for v in [20,100,30,1,.8,5,100,12,24,700,25,400,38]]; X.append([1]+x); y.append(max(5,50+3*x[5]-0.45*x[12]-0.08*x[9]-0.4*x[10]-0.5*x[0]+random.gauss(0,4)))
def solve(a,b):
 n=len(b); m=[list(map(float,a[i]))+[float(b[i])] for i in range(n)]
 for c in range(n):
  p=max(range(c,n),key=lambda r:abs(m[r][c])); m[c],m[p]=m[p],m[c]
  if abs(m[c][c])<1e-10: m[c][c]+=1e-8
  q=m[c][c]; m[c]=[v/q for v in m[c]]
  for r in range(n):
   if r!=c:
    q=m[r][c]; m[r]=[u-q*v for u,v in zip(m[r],m[c])]
 return [m[i][-1] for i in range(n)]
tr,te=split_indices(500); p=len(X[0]); A=[[sum(X[i][j]*X[i][k] for i in tr) for k in range(p)] for j in range(p)]; b=[sum(X[i][j]*y[i] for i in tr) for j in range(p)]; beta=solve(A,b); pred=[sum(v*w for v,w in zip(X[i],beta)) for i in te]; mse=sum((y[i]-q)**2 for i,q in zip(te,pred))/len(te); avg=sum(y[i] for i in te)/len(te); r2=1-sum((y[i]-q)**2 for i,q in zip(te,pred))/sum((y[i]-avg)**2 for i in te)
print('Boston Housing synthetic data:',500,'rows'); print('Features:',', '.join(names)); print('Target MEDV ($1000s)'); print('Test MSE:',round(mse,3),'R2:',round(r2,3))
