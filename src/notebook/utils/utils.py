import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
import warnings
warnings.filterwarnings("ignore")

# Dev simple functions
# Tính tỷ lệ giá trị khuyết thiếu
def calculate_missing_percentage(df=pd.DataFrame()):
    missing_value_percentage = []
    for col in df.columns:
        missing_value_percentage.append((col, df[col].isnull().sum() / len(df)))
    missing_value_percentage = sorted(missing_value_percentage, key=lambda x: x[1], reverse=True)
    missing_value_percentage = pd.DataFrame(missing_value_percentage, columns=['Feature', 'Missing Value Percentage'])
    return missing_value_percentage

# Vẽ box_plot
def plot_boxplot(df, column):
    plt.figure(figsize=(20, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()

# Vẽ histogram
def plot_histogram(df, column):
    plt.figure(figsize=(20, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
    
# Vẽ biểu đồ phân tán
def plot_scatter(df, x_column, y_column, z_column=None):
    if z_column:
        plt.figure(figsize=(20, 6))
        sns.scatterplot(x=df[x_column], y=df[y_column], hue=df[z_column])
        plt.title(f'Scatter Plot of {x_column} vs {y_column} colored by {z_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.legend(title=z_column)
        plt.show()
    else:
        if x_column == y_column:
            raise ValueError("x_column and y_column cannot be the same for a scatter plot.")

# Vẽ biểu đồ cột
def plot_bar(df, column):
    plt.figure(figsize=(20, 6))
    sns.countplot(x=df[column])
    plt.title(f'Bar Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Remove outliers using IQR
def remove_outliers_iqr(df, column, times=2.5):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr_value = iqr(df[column])
    lower_bound = q1 - times * iqr_value
    upper_bound = q3 + times * iqr_value
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Mượn các functions vẽ đồ thị
# Biểu đồ histogram
def _plot_hist_subplot(x, fieldname, bins = 10, use_kde = True):
  x = x.dropna()
  xlabel = '{} bins tickers'.format(fieldname)
  ylabel = 'Count obs in {} each bin'.format(fieldname)
  title = 'histogram plot of {} with {} bins'.format(fieldname, bins)
  ax = sns.distplot(x, bins = bins, kde = use_kde)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  return ax

# Biểu đồ barchart
def _plot_barchart_subplot(x, fieldname):
  xlabel = 'Group of {}'.format(fieldname)
  ylabel = 'Count obs in {} each bin'.format(fieldname)
  title = 'Barchart plot of {}'.format(fieldname)
  x = x.fillna('Missing')
  df_summary = x.value_counts(dropna = False)
  y_values = df_summary.values
  x_index = df_summary.index
  ax = sns.barplot(x = x_index, y = y_values, order = x_index)
  # Tạo vòng for lấy tọa độ đỉnh trên cùng của biểu đồ và thêm label thông qua annotate.
  labels = list(set(x))
  for label, p in zip(y_values, ax.patches):
    ax.annotate(label, (p.get_x()+0.25, p.get_height()+0.15))
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  return ax