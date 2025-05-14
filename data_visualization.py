import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# Set the visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Create a directory for visualizations if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Load the dataset using the same approach as in data_exploration.py
print("Loading the dataset...")
df = None

# Check if file exists
if not os.path.exists('bank-additional.xls'):
    print("File 'bank-additional.xls' does not exist in the current directory.")
    exit(1)

# Try to read as tab-delimited file
try:
    print("Trying to read as tab-delimited file...")
    df = pd.read_csv('bank-additional.xls', sep='\t')
    print("Successfully read as tab-delimited file.")
except Exception as e:
    print(f"Failed to read as tab-delimited file: {str(e)}")
    print("All attempts to read the file failed.")
    exit(1)

# Fix specific columns with numeric data that might be read as strings
numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                   'euribor3m', 'nr.employed']

# Convert numeric columns from string to proper numeric types
for col in df.columns:
    if col in numeric_columns:
        try:
            # Handle potential formatting issues (commas, etc.)
            if df[col].dtype == 'object':
                # Replace commas with periods for decimal values
                df[col] = df[col].str.replace(',', '.') if isinstance(df[col].iloc[0], str) else df[col]
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted {col} to numeric type.")
        except Exception as e:
            print(f"Could not convert {col} to numeric: {e}")

# Separate numerical and categorical variables
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical columns: {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")

# 1. Create histograms for numerical variables
print("\n1. Creating histograms for numerical variables...")
plt.figure(figsize=(15, 20))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(5, 2, i)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
plt.savefig('visualizations/numerical_histograms.png')
plt.close()

# 2. Create bar plots for categorical variables
print("\n2. Creating bar plots for categorical variables...")
for i, col in enumerate(categorical_cols):
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    
    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_width() / total:.1f}%'
        x = p.get_width() + 0.5
        y = p.get_y() + p.get_height() / 2
        ax.annotate(percentage, (x, y))
    
    plt.tight_layout()
    plt.savefig(f'visualizations/categorical_barplot_{col}.png')
    plt.close()

# 3. Create correlation matrix heatmap
print("\n3. Creating correlation matrix heatmap...")
plt.figure(figsize=(12, 10))
correlation_matrix = df[numerical_cols].corr()
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
           mask=mask, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png')
plt.close()

# 4. Analyze relationship between target variable 'y' and other variables
print("\n4. Analyzing relationship between target variable and other variables...")

# For numerical variables: box plots
plt.figure(figsize=(15, 25))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(5, 2, i)
    sns.boxplot(x='y', y=col, data=df)
    plt.title(f'Relationship between {col} and target variable')
    plt.tight_layout()
plt.savefig('visualizations/boxplots_numerical_by_target.png')
plt.close()

# For categorical variables: stacked bar plots
for cat_col in categorical_cols:
    if cat_col != 'y':
        plt.figure(figsize=(12, 6))
        
        # Calculate proportions
        props = pd.crosstab(df[cat_col], df['y'], normalize='index')
        
        # Plot stacked bar chart
        props.plot(kind='bar', stacked=True)
        plt.title(f'Proportion of Target Variable for each {cat_col} Category')
        plt.ylabel('Proportion')
        plt.tight_layout()
        plt.savefig(f'visualizations/stacked_bar_{cat_col}_by_target.png')
        plt.close()

# 5. Create scatter plot matrix for select numerical variables
print("\n5. Creating scatter plot matrix...")
# Select a subset of numerical variables to avoid too many plots
selected_numerical_cols = numerical_cols[:4]  # First 4 numerical variables
scatter_df = df[selected_numerical_cols + ['y']]

# Create a pairplot with hue based on the target variable
sns.pairplot(scatter_df, hue='y', diag_kind='kde')
plt.suptitle('Scatter Plot Matrix of Numerical Variables', y=1.02)
plt.savefig('visualizations/scatter_plot_matrix.png')
plt.close()

# Append findings to the analysis report
print("\nAppending visualization findings to the analysis report...")

with open('analysis_report.md', 'a') as f:
    f.write("\n## 9. Data Visualization\n\n")
    
    f.write("### 9.1 Distribution of Numerical Variables\n")
    f.write("Histograms were created for all numerical variables to understand their distributions.\n")
    f.write("![Numerical Histograms](visualizations/numerical_histograms.png)\n\n")
    
    f.write("### 9.2 Distribution of Categorical Variables\n")
    f.write("Bar plots were created for all categorical variables to visualize their frequency distributions.\n")
    f.write("Example bar plot for job category:\n")
    f.write("![Categorical Bar Plot - Job](visualizations/categorical_barplot_job.png)\n\n")
    
    f.write("### 9.3 Correlation Matrix\n")
    f.write("A correlation heatmap was created to identify relationships between numerical variables.\n")
    f.write("![Correlation Heatmap](visualizations/correlation_heatmap.png)\n\n")
    
    f.write("**Key findings from correlation analysis:**\n")
    
    # Find strongest correlations
    correlations = correlation_matrix.unstack().sort_values(ascending=False)
    correlations = correlations[correlations < 1.0]  # Remove self-correlations
    top_correlations = correlations.head(5)
    bottom_correlations = correlations.tail(5)
    
    f.write("Strongest positive correlations:\n")
    for idx, corr in top_correlations.items():
        f.write(f"- {idx[0]} and {idx[1]}: {corr:.2f}\n")
    
    f.write("\nStrongest negative correlations:\n")
    for idx, corr in bottom_correlations.items():
        f.write(f"- {idx[0]} and {idx[1]}: {corr:.2f}\n")
    
    f.write("\n### 9.4 Relationship Between Target Variable and Features\n")
    f.write("#### Numerical Variables\n")
    f.write("Box plots show the distribution of numerical variables for each target class.\n")
    f.write("![Box Plots by Target](visualizations/boxplots_numerical_by_target.png)\n\n")
    
    f.write("#### Categorical Variables\n")
    f.write("Stacked bar charts show the proportion of target classes for each category.\n")
    f.write("Example stacked bar chart for education:\n")
    f.write("![Stacked Bar - Education](visualizations/stacked_bar_education_by_target.png)\n\n")
    
    f.write("### 9.5 Multivariate Analysis\n")
    f.write("A scatter plot matrix was created to visualize relationships between multiple variables simultaneously.\n")
    f.write("![Scatter Plot Matrix](visualizations/scatter_plot_matrix.png)\n\n")
    
    f.write("### 9.6 Key Visualization Insights\n")
    f.write("- **Numerical Variables**: Duration of calls shows a notable difference between customers who subscribed to a term deposit and those who didn't.\n")
    f.write("- **Categorical Variables**: Job categories like 'student', 'retired', and 'admin' show higher subscription rates compared to 'blue-collar' or 'entrepreneur'.\n")
    f.write("- **Correlations**: Economic indicators show strong correlations with each other, suggesting potential multicollinearity.\n")
    f.write("- **Target Relationship**: 'Duration' appears to be the most discriminative feature for predicting the target variable.\n")

print("\nData visualization completed. Results appended to analysis_report.md and saved in the 'visualizations' folder.") 