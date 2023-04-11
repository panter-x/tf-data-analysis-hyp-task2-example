import pandas as pd
import numpy as np
#from scipy.stats import anderson_ksamp
from scipy.stats import ks_2samp 

chat_id = 46951859 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    alpha = 0.01
    
    #return anderson_ksamp([x, y]).pvalue < alpha
    pv = ks_2samp(x, y)[1]
    return pv < alpha
