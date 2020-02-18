import numpy as np
from util.thresholding import *

def altProjNiave(M,r,s, fTol=1e-10, maxIter=1000):
    res=np.inf
    L=np.zeros(M.shape)
    S=np.zeros(M.shape)
    for k in range(0, maxIter):
        L = lowRankProj(M-S, r)
        S = sparseProj(M-L, s)
        res0 = res
        res = np.linalg.norm(M-(L+S), ord='fro')
        if (res0-res)/res < fTol:
            break
    return (L,S)


def altSoftThresh(M, beta,fTol=1e-6, maxIter=1000):
    sqrtN = np.sqrt(min(M.shape))
    res = np.inf
    L=np.zeros(M.shape)
    S=np.zeros(M.shape)
    # TODO: Make beta increase with every iteration
    for k in range(0, maxIter):
        S = sparseSoftThresholding(M-L, 1/(sqrtN*beta))
        L = lowRankSoftThresholding(M-S, 1/beta)
        # TODO: fix the stopping criteria in altSoftThresh
      #  res0 = res
       # res = np.linalg.norm(M-(L+S), ord='fro')
       # if (res0-res)/res < fTol:
       #     break
    return(L,S)


if __name__=="__main__":
    L=np.ones([10,10])
    S=np.zeros([10,10])
    S[1,2]=10
    S[5,9]=10
    S[3,5]=-10
    M=L+S

    (Lproj,Sproj)=altProjNiave(M,1,3)
    print(Lproj)
    print(Sproj)
