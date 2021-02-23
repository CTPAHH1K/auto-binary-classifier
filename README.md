# basic binary classification supervised auto-ml library

This lib is intended to automate supervised binary classification

Usage is as simple as

```python
from automl.binary_classifier import BinaryClassifier

cls = BinaryClassifier()  
cls.fit(X_train, y_train)  
pred = cls.predict(X_test)
```