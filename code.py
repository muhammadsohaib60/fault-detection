# %%
# %pip install joblib matplotlib scikit-learn seaborn tensorflow
# %pip install scikeras ray
# %pip install "ray[train]"

# %%
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from statistics import mode
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation, LSTM, GRU, SimpleRNN, Conv1D, TimeDistributed, MaxPooling1D, Flatten, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Hyperparameter
import ray
from ray import train, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air.integrations.keras import ReportCheckpointCallback

from tensorflow.keras.optimizers import Adam, RMSprop
from functools import partial
from ray.tune.schedulers import ASHAScheduler


# Setting the random seeds for reproductibility
np.random.seed(42)
random.seed(42)

# %% [markdown]
# # Read Data Files

# %%

# File paths
file_path_healthy = './HIL Data 062024/ACC faults different scenarios/Sc5Healthy.csv'
file_path_delay = './HIL Data 062024/ACC faults different scenarios/Delay_APP_50.csv'
file_path_gain = './HIL Data 062024/RPM  faults different scenarios/FP75_gain.csv'
file_path_noise = './HIL Data 062024/ACC faults different scenarios/Sc5NoiseAPP.csv'
file_path_loss = './HIL Data 062024/ACC faults different scenarios/Sc5PacketLossAPP.csv'

# %%

# Function to parse the general information section


def parse_general_info(lines):
    general_info = {}
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 3:
            category, key, value = parts[0], parts[1], parts[2]
            if category == "General":
                general_info[key] = value
    return general_info

# Function to parse the trace information section


def parse_trace_info(lines):
    trace_info = {}
    headers = lines[0].split('\t')
    for line in lines[1:]:
        values = line.split('\t')
        for i, header in enumerate(headers):
            if i < len(values):
                if header not in trace_info:
                    trace_info[header] = []
                trace_info[header].append(values[i])
    return trace_info

# Function to parse the trace values section


def parse_trace_values(lines):
    trace_values = [line.split(',') for line in lines]
    df = pd.DataFrame(trace_values[1:], columns=trace_values[0])
    return df

# Read the text data


def read_this(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse sections
    general_info_lines = []
    trace_info_lines = []
    trace_values_lines = []
    current_section = None

    for line in lines:
        line = line.strip()
        if 'descriptions' in line:
            current_section = 'general_info'
        elif 'trace_size' in line:
            current_section = 'trace_info'
        elif 'trace_values' in line:
            current_section = 'trace_values'
        elif line:
            if current_section == 'general_info':
                general_info_lines.append(line)
            elif current_section == 'trace_info':
                trace_info_lines.append(line)
            elif current_section == 'trace_values':
                trace_values_lines.append(line)

    # Process sections
    general_info = parse_general_info(general_info_lines)
    trace_info = parse_trace_info(trace_info_lines)
    trace_values = parse_trace_values(trace_values_lines)

    return general_info, trace_info, trace_values


# %% [markdown]
# # Labeling

# %%
# Read data
healthy_data = read_this(file_path_healthy)[2]
delay_app = read_this(file_path_delay)[2]
gain_rpm = read_this(file_path_gain)[2]
noise_app = read_this(file_path_noise)[2]
packetloss_app = read_this(file_path_loss)[2]

healthy_data = healthy_data.values.flatten()
delay_app = delay_app.values.flatten()
gain_rpm = gain_rpm.values.flatten()
noise_app = noise_app.values.flatten()
packetloss_app = packetloss_app.values.flatten()


# %%
def clean_and_convert(arr):
    # Convert to numeric data, skipping empty strings
    cleaned_arr = np.array([float(x) if isinstance(
        x, str) and x.strip() else x for x in arr])
    return cleaned_arr


healthy_data = clean_and_convert(healthy_data)
delay_app = clean_and_convert(delay_app)
gain_rpm = clean_and_convert(gain_rpm)
noise_app = clean_and_convert(noise_app)
packetloss_app = clean_and_convert(packetloss_app)


healthy_data = clean_and_convert(healthy_data)
delay_app = clean_and_convert(delay_app)
gain_rpm = clean_and_convert(gain_rpm)
noise_app = clean_and_convert(noise_app)
packetloss_app = clean_and_convert(packetloss_app)

# %%


def calculate_diff_and_label(healthy, data, threshold_divisor):
    diff = np.abs(healthy - data)
    threshold = np.max(healthy, axis=0) / threshold_divisor
    label = (diff > threshold).astype(int)
    return label

# %%


label_gain = calculate_diff_and_label(healthy_data, gain_rpm, 3.5)
label_delay = calculate_diff_and_label(healthy_data, delay_app, 3.5)
label_loss = calculate_diff_and_label(healthy_data, packetloss_app, 3.5)
label_noise = calculate_diff_and_label(healthy_data, noise_app, 3.5)

# Add labels to the datasets
healthy_data['delay_class'] = 0
healthy_data['gain_class'] = 0
healthy_data['noise_class'] = 0
healthy_data['loss_class'] = 0

delay_app['delay_class'] = label_delay.max(axis=1)
delay_app['gain_class'] = 0
delay_app['noise_class'] = 0
delay_app['loss_class'] = 0

gain_rpm['gain_class'] = label_gain.max(axis=1)
gain_rpm['delay_class'] = 0
gain_rpm['noise_class'] = 0
gain_rpm['loss_class'] = 0

noise_app['noise_class'] = label_noise.max(axis=1)
noise_app['delay_class'] = 0
noise_app['gain_class'] = 0
noise_app['loss_class'] = 0

packetloss_app['loss_class'] = label_loss.max(axis=1)
packetloss_app['delay_class'] = 0
packetloss_app['gain_class'] = 0
packetloss_app['noise_class'] = 0

# %%

# Align columns before combining
all_dataframes = [healthy_data, delay_app, gain_rpm, noise_app, packetloss_app]
all_columns = list(set().union(*[df.columns for df in all_dataframes]))
for df in all_dataframes:
    df = df.reindex(columns=all_columns, fill_value=np.nan)

# Combine data
combined_data = pd.concat(all_dataframes, ignore_index=True)

combined_data

# %%
# Prepare the data for logistic regression
X = combined_data.iloc[:, 1:].values
y = combined_data[['delay_class', 'gain_class',
                   'noise_class', 'loss_class']].values
