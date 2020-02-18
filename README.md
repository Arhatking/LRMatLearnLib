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


## Using Robust Principle Component Analysis for Background Forground Seperation

Import packages, open the video file, and flatten the frames into vectors.  

```python
import sys
sys.path.append('../')
from datasets.data_loader import *
from RPCA.algorithms import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

v=load_video("../datasets/videos/escalator.avi")
(n_frames,d1,d2)=v.shape
v=v.reshape(n_frames, d1*d2)


```
run altProjNiave, a basic RPCA algorithm.  The first arguement is the matrix to be decomposed, the second is the rank of the low rank matrix, and the third is the number of entries in the sparse matrix.

```python
(L,S)=altProjNiave(v, 2,100*n_frames)
```
Reshape the frames back into images and plot them. 

```python
L=L.reshape(n_frames,d1,d2)
S=S.reshape(n_frames,d1,d2)
v=v.reshape(n_frames,d1,d2)
all=np.concatenate((v,L,S), axis=2)

plt.imshow(all[1,:,:])
plt.show()
```
![png](escalator.png)

