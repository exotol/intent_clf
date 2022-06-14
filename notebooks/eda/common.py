import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from wordcloud import STOPWORDS, ImageColorGenerator, WordCloud

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 50)
pd.set_option("display.max_colwidth", None)


PROJECT_PATH_ROOT = os.getenv("PROJECT_PATH_ROOT")
SEED = int(os.getenv("SEED"))

DATA_PATH = os.path.join(PROJECT_PATH_ROOT, "data", "raw")

TRAIN_PATH = os.path.join(DATA_PATH, "train.xlsx")
TEST_PATH = os.path.join(DATA_PATH, "test.xlsx")
