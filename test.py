import pandas as pd
import numpy as np
X = 1
Y = 2
logic = (X>Y)
pd.DataFrame(np.where(logic, X, Y))