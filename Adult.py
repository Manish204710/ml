"""Experiment 3: Adult income decision tree with small depth comparison."""
from experiments_common import make_adult,split_indices,tree_fit,tree_predict,metrics
X,y=make_adult(); tr,te=split_indices(500,seed=42); print('Adult-style synthetic data:',500,'rows; target 1=>50K, 0=<=50K'); print('Features: age, education-num, hours/week, capital-gain, workclass-code, occupation-code, married, sex')
for depth in (3,5,8):
 t=tree_fit([X[i] for i in tr],[y[i] for i in tr],depth,5); p=[tree_predict(t,X[i]) for i in te]; print('Decision tree max_depth=',depth,metrics([y[i] for i in te],p))
