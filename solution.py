import pandas as pd
import numpy as np


chat_id = 46951859 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    alpha = 0.01
    
    return anderson_ksamp([x, y]).pvalue < alpha
