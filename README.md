# LRMatLearnLib

## Using Matrix Completion for Movie Recomendations

Start by loading the MovieLens1M dataset

```python
data = np.loadtxt( 'ml-1m/ratings.dat',delimiter='::' )
X=data[:, [0,1]].astype(int)-1
y=data[:,2]

n_users=max(X[:,0])+1
n_movies=max(X[:,1])+1

print((n_users,n_movies))
```

    (6040, 3952)


So, we have 6040 users and 3952 movies.  That's a total of about 23 million potential ratings, of which we know 1 million.  We're going to reserve 200,000 of the ratings to test our results.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Import MC.py, fit the data and make predictions

```python
from MC import *
from statistics import mean

mc_model=MC(n_users,n_movies,5)
mc_model.fit(np.array(X_train).transpose(), y_train)
y_predict=mc_model.predict((np.array(X_test).transpose()))

print("MAE:",mean(abs(y_test-y_predict)))

print("Percent of predictions off my less than 1: ",np.sum(abs(y_test-y_predict)<1)/len(y_test))
```

    MAE: 0.6910439339771605
    Percent of predictions off my less than 1:  0.7603603243318903

