# Convolutional Decision Trees

Code for convolutional decision trees (CDT).

Used in the following papers:
1. [Boosted Convolutional Decision Trees for Translationally Invariant Pattern Recognition and Transfer Learning](http://www.ccsenet.org/journal/index.php/ijsp/article/view/0/38163)
2. [Local genomic features predict the distinct and overlapping binding patterns of the bHLH-Zip family oncoproteins MITF and MYC-MAX.](https://www.ncbi.nlm.nih.gov/pubmed/30548162)


## Examples

#### Classification CDT for image classification
```python
import Optim as cdtopt
import Loss as cdtloss
import CDT as cdt

DNA=False
filter_dimensions = (8,8)
cdt_max_depth = 3

optimizer = cdtopt.CEOptimizer(cdtloss.child_entropy, filter_dimensions, DNA=DNA)
dt = cdt.CDTClassifier(cdt_max_depth, filter_dimensions, DNA=DNA, optimizer=optimizer)
dt.fit(X, y)
dt.predict(X)
```


#### Regression CDT for DNA sequences
```python
DNA = True
input_sequence_length = 200
filter_length = 8
cdt_max_depth = 2

optimizer = cdtopt.CEOptimizer(cdtloss.child_variance, filter_length, input_sequence_length, DNA=DNA)
dt = cdt.CDTRegressor(cdt_max_depth, filter_length, input_sequence_length, DNA=DNA, optimizer=optimizer)
dt.fit(X, y)
dt.predict(X)
```


#### AdaBoosted CDT
```python
from sklearn.ensemble import AdaBoostClassifier

DNA=False
filter_dimensions = (8,8)
cdt_max_depth = 3

optimizer = cdtopt.CEOptimizer(cdtloss.child_entropy, filter_dimensions, DNA=DNA)
dt = cdt.CDTClassifier(cdt_max_depth, filter_dimensions, DNA=DNA, optimizer=optimizer)

bdt = AdaBoostClassifier(dt, n_estimators=20)
bdt.fit(X, y)
bdt.predict(X)
```

#### Gradient Boosted CDT for DNA sequences
```python
from GradientBoostedCDT import GradientBoostedCDT

DNA = True
input_sequence_length = 200
filter_length = 8
cdt_max_depth = 2
num_estimators = 30

optimizer = cdtopt.CEOptimizer(cdtloss.child_variance, filter_length, input_sequence_length, DNA=DNA)
dt = cdt.CDTRegressor(cdt_max_depth, filter_length, input_sequence_length, DNA=DNA, optimizer=optimizer)
bdt = GradientBoostedCDT(dt, num_estimators)

bdt.fit(X,y)
bdt.decision_function(X)
```

