import numpy as np
from math import *
from scipy.sparse import coo_matrix
def lmafit_mc_adp(m,n,r,known,data):
    L = len(data);

    # set parameters
    tol = 1e-5;
    maxit = 500;
    iprint = 2;
    est_rank = 1;
    rank_max = max(floor(0.1 * min(m, n)), 2 * r);
    rank_min = 1;
    rk_jump = 10;
    init = 0;
    save_res = 0;


    #Initialize Variables

    X = np.random.rand(m, r)
    Y = np.random.rand(n, r)
    Res=data-partXY(X, Y, known)
    S = coo_matrix((Res, known), shape=(m,n))
    #S=np.zeros((m,n))
    alf=0
    increment=1

    Res = data - partXY(X, Y, known)

    res = np.linalg.norm(Res);

    #main loop

    for iter in range(0, maxit):
        X0=X
        Y0=Y
        Res0=Res
        res0=res

        X=X+ S.dot(Y).dot(np.linalg.pinv(Y.transpose().dot(Y)))
        XXInv=np.linalg.pinv(X.transpose().dot(X));
        Y=Y.dot(X0.transpose().dot(X)).dot(XXInv)+S.transpose().dot(X).dot(XXInv)
        Res=data-partXY(X, Y, known)

        res = np.linalg.norm(Res);
        ratio = res / res0;

        if ratio >= 1:
            increment = max(0.1 * alf, 0.1 * increment)
            X = X0
            Y = Y0
            Res = Res0
            res = res0
            alf = 0
        elif ratio > 0.7:
            increment = max(increment, 0.25 * alf)
            alf = alf + increment

        S = coo_matrix(((alf+1)*Res, known))

    return (X,Y)


def partXY(x,y, known):
    return np.sum(np.multiply(x[known[0][:], :],y[known[1][:],:]), axis=1)


if __name__ == "__main__":
    known = [[0, 0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 3, 0, 1, 2, 0, 1, 3]]
    y = [1, 1, 2, 2, 2, 1, 3, 1, 2, 1]
    m=3
    n=4
    r=2

    (U,V) = lmafit_mc_adp(3, 4, 2, known, y)
    print(U.dot(V.transpose()))
