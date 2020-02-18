import numpy as np
from util.thresholding import *

def altProjNiave(M,r,s):
    for iter in range(0, maxIter):
        L=lowRankProj(M-S, r)
        S=sparseProj(M-L, s)
    return (L,S)


def altThresh(M, beta):
    for iter in range(0, maxIter):

    return(L,S)

