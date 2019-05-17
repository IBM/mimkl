from ._pymimkl import *
# importing wrapped models
from .average_mkl import AverageMKL
from .easy_mkl import EasyMKL
from .umkl_knn import UMKLKNN

# cleanup unused objects
del(EasyMKL_)
del(UMKLKNN_)
del(AverageMKL_)
