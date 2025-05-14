import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Load the dataset
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
    # Try other approaches if tab-delimited reading fails
    try:
        # Try reading as semicolon-separated CSV
        print("Trying to read as semicolon-separated CSV...")
        df = pd.read_csv('bank-additional.xls', sep=';')
        print("Successfully read as semicolon-separated CSV.")
    except Exception as e:
        print(f"Failed to read as semicolon-separated CSV: {str(e)}")
        try:
            # Try with xlrd engine
            print("Trying with xlrd engine...")
            df = pd.read_excel('bank-additional.xls', engine='xlrd')
            print("Successfully read with xlrd engine.")
        except Exception as e:
            print(f"Failed with xlrd engine: {str(e)}")
            print("All attempts to read the file failed.")
            exit(1)

# Print first few rows to confirm data was loaded correctly
print("\nFirst 5 rows of the dataset:")
print(df.head())

# If the data has only one column (which contains all columns concatenated),
# we need to split it into proper columns
if df.shape[1] == 1:
    print("\nData appears to be in a single column. Attempting to split into proper columns...")
    # Get the first column name
    first_col = df.columns[0]
    
    # Check for different delimiters in the content
    sample_row = df.iloc[0, 0]
    print(f"Sample row: {sample_row}")
    
    if ';' in sample_row:
        # Split on semicolons
        df = df[first_col].str.split(';', expand=True)
        print("Split on semicolons.")
    elif ',' in sample_row:
        # Split on commas
        df = df[first_col].str.split(',', expand=True)
        print("Split on commas.")
    else:
        print("Could not determine proper delimiter for splitting the data.")
        exit(1)
    
    # Read the first row as header
    header = df.iloc[0].values
    df = df[1:]
    df.columns = header
    print("\nAfter splitting data into columns:")
    print(df.head())

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
    
print("\nAfter converting numeric columns:")
print(df.dtypes)

# 2. Understand the data structure
print("\n2. Data Structure:")
print(f"Shape: {df.shape} (rows, columns)")
print("\nColumn data types:")
print(df.dtypes)

# 3. Detect missing values
print("\n3. Missing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.any() > 0 else "No missing values found in the dataset.")

# Check for 'unknown' values in each column
print("\nColumns with 'unknown' values:")
unknown_counts = {}
for col in df.columns:
    if df[col].dtype == 'object':
        unknown_count = (df[col] == 'unknown').sum()
        if unknown_count > 0:
            unknown_counts[col] = unknown_count

if unknown_counts:
    unknown_df = pd.DataFrame.from_dict(unknown_counts, orient='index', columns=['Count'])
    unknown_df['Percentage'] = unknown_df['Count'] / len(df) * 100
    print(unknown_df.sort_values('Count', ascending=False))
else:
    print("No 'unknown' values found in the dataset.")

# 4. Separate numerical and categorical variables
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\n4. Variable Types:")
print(f"Numerical variables ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical variables ({len(categorical_cols)}): {categorical_cols}")

# 5. Compute summary statistics
print("\n5. Summary Statistics for Numerical Variables:")
if numerical_cols:
    print(df[numerical_cols].describe())
else:
    print("No numerical columns found.")

print("\nSummary Statistics for Categorical Variables:")
for cat_col in categorical_cols:
    print(f"\n{cat_col} value counts:")
    value_counts = df[cat_col].value_counts()
    print(value_counts)
    print(f"Percentage distribution:")
    print((value_counts / len(df) * 100).round(2))

# 6. Analyze the distribution of the target variable 'y'
if 'y' in df.columns:
    print("\n6. Target Variable Analysis:")
    target_counts = df['y'].value_counts()
    print("Target variable counts:")
    print(target_counts)
    print("\nTarget variable distribution (%):")
    print((target_counts / len(df) * 100).round(2))

    # Check class imbalance
    imbalance_ratio = target_counts.max() / target_counts.min()
    print(f"\nClass imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")
else:
    print("\n6. Target Variable Analysis:")
    print("Target variable 'y' not found in the dataset.")

# Save insights to a file
print("\nSaving data exploration results...")
with open('analysis_report.md', 'w') as f:
    f.write("# Data Understanding and Exploration Report\n\n")
    
    f.write("## 1. Data Overview\n")
    f.write("This report summarizes the initial exploration of the bank marketing dataset. The dataset contains information about bank clients, marketing campaign details, and whether clients subscribed to a term deposit (target variable 'y').\n\n")
    
    f.write("## 2. Data Structure\n")
    f.write(f"- **Number of records (rows)**: {df.shape[0]}\n")
    f.write(f"- **Number of features (columns)**: {df.shape[1]}\n")
    f.write("- **Data types**:\n")
    for col, dtype in df.dtypes.items():
        f.write(f"  - {col}: {dtype}\n")
    f.write("\n")
    
    f.write("## 3. Missing Values\n")
    if missing_values.any() > 0:
        f.write("- **Traditional missing values**:\n")
        for col, count in missing_values[missing_values > 0].items():
            f.write(f"  - {col}: {count} ({count/len(df)*100:.2f}%)\n")
    else:
        f.write("- **Traditional missing values**: None found\n")
    
    if unknown_counts:
        f.write("- **'Unknown' values**:\n")
        for col, count in unknown_counts.items():
            f.write(f"  - {col}: {count} ({count/len(df)*100:.2f}%)\n")
    else:
        f.write("- **'Unknown' values**: None found\n")
    f.write("\n")
    
    f.write("## 4. Variable Types\n")
    f.write(f"- **Numerical variables ({len(numerical_cols)})**: {', '.join(numerical_cols)}\n")
    f.write(f"- **Categorical variables ({len(categorical_cols)})**: {', '.join(categorical_cols)}\n\n")
    
    f.write("## 5. Summary Statistics\n")
    f.write("### Numerical Variables\n")
    if numerical_cols:
        stats = df[numerical_cols].describe().round(2)
        f.write("| Statistic | " + " | ".join(stats.columns) + " |\n")
        f.write("| --- | " + " | ".join(["---" for _ in stats.columns]) + " |\n")
        for idx, row in stats.iterrows():
            f.write(f"| {idx} | " + " | ".join([str(val) for val in row.values]) + " |\n")
    else:
        f.write("No numerical variables found.\n")
    f.write("\n")
    
    f.write("### Categorical Variables\n")
    for cat_col in categorical_cols[:5]:  # Limit to first 5 categorical variables to keep report concise
        f.write(f"#### {cat_col}\n")
        value_counts = df[cat_col].value_counts()
        f.write("| Value | Count | Percentage |\n")
        f.write("| --- | --- | --- |\n")
        for val, count in value_counts.items():
            f.write(f"| {val} | {count} | {count/len(df)*100:.2f}% |\n")
        f.write("\n")
    
    if 'y' in df.columns:
        f.write("## 6. Target Variable Analysis\n")
        f.write("### Distribution of Target Variable 'y'\n")
        target_counts = df['y'].value_counts()
        f.write("| Value | Count | Percentage |\n")
        f.write("| --- | --- | --- |\n")
        for val, count in target_counts.items():
            f.write(f"| {val} | {count} | {count/len(df)*100:.2f}% |\n")
        f.write(f"\n**Class imbalance ratio (majority:minority)**: {imbalance_ratio:.2f}:1\n\n")
    
    f.write("## 7. Initial Insights\n")
    f.write("- The dataset contains information about bank marketing campaigns.\n")
    if 'y' in df.columns:
        maj_class = target_counts.idxmax()
        maj_pct = target_counts.max() / len(df) * 100
        f.write(f"- The target variable shows {'a balanced' if imbalance_ratio < 3 else 'an imbalanced'} distribution with the majority class '{maj_class}' representing {maj_pct:.2f}% of the data.\n")
    if unknown_counts:
        f.write(f"- Several columns contain 'unknown' values, which will need to be addressed in the data preprocessing step.\n")
    
    f.write("\n## 8. Next Steps\n")
    f.write("- **Data preprocessing strategy**:\n")
    if unknown_counts:
        f.write("  - Handle 'unknown' values through imputation or encoding\n")
    f.write("  - Convert categorical variables to numerical representations (one-hot encoding or label encoding)\n")
    f.write("  - Apply feature scaling for numerical variables\n")
    if 'y' in df.columns and imbalance_ratio > 3:
        f.write("  - Address class imbalance using techniques like SMOTE, class weighting, or resampling\n")
    f.write("- **Feature engineering**:\n")
    f.write("  - Consider creating new features based on domain knowledge\n")
    f.write("  - Apply feature selection or dimensionality reduction techniques\n")
    f.write("- **Model development**:\n")
    f.write("  - Prepare training and test datasets\n")
    f.write("  - Implement various classification algorithms\n")
    f.write("  - Tune hyperparameters\n")
    f.write("  - Evaluate model performance\n")

print("\nData exploration completed. See analysis_report.md for the summary.") 