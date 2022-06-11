import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from tqdm import tqdm

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', None)


PROJECT_PATH_ROOT = os.getenv('PROJECT_PATH_ROOT')
SEED = int(os.getenv('SEED'))

DATA_PATH = os.path.join(PROJECT_PATH_ROOT, "data", "raw")

TRAIN_PATH = os.path.join(DATA_PATH, "train.xlsx")
TEST_PATH = os.path.join(DATA_PATH, "test.xlsx")