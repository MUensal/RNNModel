import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# Reading csv_file: skips header, separation with delimiter, encoding needs to be set here
df = pd.read_csv(r'Data_prepared.csv',
                 encoding='ISO-8859-1',
                         engine='python',
                         delimiter=';', header=0)


print(df.head())



