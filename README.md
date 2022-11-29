# We’ll be using the digits dataset in the scikit learn library to predict digit values from images using the logistic regression model in Python.

### Importing libraries and their associated methods


```python
import re
```


```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

%matplotlib inline
```


```python
digits = load_digits()
```

### Determining the total number of images and labels


```python
print("Image Data Shape", digits.data.shape)
print("Label Data Shape", digits.target.shape)
```

    Image Data Shape (1797, 64)
    Label Data Shape (1797,)


### Displaying some of the images and their labels


```python
plt.figure(figsize =(10,10))

for index, (image, label) in enumerate (zip(digits.data[0:20], digits.target[0:20])):
    plt.subplot(4, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap = 'magma')
    plt.title('Training: %i\n' % label, fontsize = 12)
```


![output_8_0](https://user-images.githubusercontent.com/50436546/204611817-edb2c1bc-262b-41f0-9756-5ae6e2028c17.png)


### Dividing dataset into “training” and “test” set 


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (digits.data, digits.target, test_size = 0.23, random_state = 2)
```


```python
print(x_train.shape)
```

    (1383, 64)



```python
print(y_train.shape)
```

    (1383,)



```python
print(x_test.shape)
```

    (414, 64)



```python
print(y_test.shape)
```

    (414,)


### Importing the logistic regression model


```python
from sklearn.linear_model import LogisticRegression
```

### Making an instance of the model and training it

Increased max_iter to 10000 as default value is 1000. Possibly, increasing no. of iterations will help algorithm to converge. For me it converged and solver was -'lbfgs'


```python
logisticRegr = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000)
```


```python
logisticRegr.fit(x_train, y_train)
```




    LogisticRegression(class_weight='balanced', max_iter=10000)



### Predicting the output of the first element of the test set


```python
print(logisticRegr.predict(x_test[0].reshape(1,-1)))
```

    [4]


### Predicting the output of the first 10 elements of the test set


```python
logisticRegr.predict(x_test[0:10])
```




    array([4, 0, 9, 1, 8, 7, 1, 5, 1, 6])



### Prediction for the entire dataset


```python
x_pred = logisticRegr.predict(x_test)
```

### Determining the accuracy of the model


```python
score = logisticRegr.score(x_test, y_test)
score
```




    0.9492753623188406



### Representing the confusion matrix in a heat map


```python
from sklearn.metrics import confusion_matrix

cf_matrix = confusion_matrix(y_test, x_pred)
print(cf_matrix)
```

    [[37  0  0  0  1  0  0  0  0  0]
     [ 0 46  0  1  0  0  0  0  1  0]
     [ 0  0 43  0  0  0  0  0  0  0]
     [ 0  0  0 39  0  0  0  2  1  0]
     [ 0  0  0  0 34  0  0  0  3  1]
     [ 0  1  0  0  1 43  0  0  0  1]
     [ 0  1  0  0  0  0 39  0  1  0]
     [ 0  0  0  0  0  0  0 45  1  0]
     [ 0  0  0  0  1  0  0  0 36  1]
     [ 0  0  0  1  0  1  0  0  1 31]]



```python
import seaborn as sns

sns.heatmap(cf_matrix, annot=True, cmap = 'magma')
```




    <AxesSubplot:>


![output_31_1](https://user-images.githubusercontent.com/50436546/204611738-6cd6bcb1-ef50-469b-88ca-002f7b2ca26f.png)



### Presenting predictions and actual output


```python
index = 0
misclassifiedIndex = []
```


```python
for predict, actual in zip(x_pred, y_test):
    if predict == actual:
        misclassifiedIndex.append(index)
    index += 1

plt.figure(figsize=(16,14))

for plotIndex, wrong in enumerate(misclassifiedIndex[0:20]):
    plt.subplot(4,5, plotIndex + 1)
    plt.imshow(np.reshape(x_test[wrong], (8,8)), cmap = 'magma')
    plt.title("Predicted: {}, Actual: {}" .format(predictions[wrong], y_test[wrong]), fontsize = 14)
```
    
![output_34_0](https://user-images.githubusercontent.com/50436546/204611717-3c9bea0f-1386-4dc6-ab23-7f01ee02542c.png)



```python

```
