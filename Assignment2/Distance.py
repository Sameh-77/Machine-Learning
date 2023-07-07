#======================================================#
#== Sameh Algharabli - CNG 409 -- Assignment2 -- Distance ==#
#======================================================#
#--- Libraries ---#
import numpy as np
from numpy.linalg import norm
from math import *
#------------------------------------------------------#

class Distance:
    @staticmethod
    def calculateCosineDistance(x, y):
        cosine = np.dot(x, y)/(norm(x)*norm(y))
        #print(cosine)
        return (1-cosine)


    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        summation = sum((pow(abs(a-b), p)) for a, b in zip(x, y))
        power = 1/float(p)
        minkowski = pow(summation,power)
        return minkowski
    @staticmethod
    def calculateMahalanobisDistance(x, y, S_minus_1):
        mahalanobis = np.sqrt(np.dot(np.dot((x-y), S_minus_1), (x-y)))
        #print(mahalanobis)
        return (mahalanobis)

