"""Experiment 4: Adult income random forest from standard-library trees."""
from experiments_common import make_adult,split_indices,forest_fit,forest_predict,metrics
X,y=make_adult(); tr,te=split_indices(500,seed=42); model=forest_fit([X[i] for i in tr],[y[i] for i in tr],count=31,depth=6,leaf=4); p=[forest_predict(model,X[i]) for i in te]
print('Adult-style synthetic data:',500,'rows; target 1=>50K, 0=<=50K'); print('Random forest: 31 bootstrap trees, random sqrt(feature count) per tree'); print(metrics([y[i] for i in te],p))
