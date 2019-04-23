# Convolutional Decision Trees

Code for convolutional decision trees (CDT).

Used in the following papers:
[Boosted Convolutional Decision Trees for Translationally Invariant Pattern Recognition and Transfer Learning](http://www.ccsenet.org/journal/index.php/ijsp/article/view/0/38163)
[Local genomic features predict the distinct and overlapping binding patterns of the bHLH-Zip family oncoproteins MITF and MYC-MAX.](https://www.ncbi.nlm.nih.gov/pubmed/30548162)


## Examples

```python
import Optim as cdtopt
import Loss as cdtloss
import CDT as cdt

optimizer = cdtopt.CEOptimizer(cdtloss.child_entropy, 8, 200, DNA=True)
DT = cdt.CDTClassifier(2, 8, 200, DNA=True, optimizer=optimizer)

DT.fit(X, y)

DT.predict(X)
```
