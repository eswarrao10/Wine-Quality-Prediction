import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"C:\Users\rahil\Downloads\wine_data_preprocessed.csv"
df = pd.read_csv(file_path)

# Summary statistics
summary_stats = df.describe().T

# Outlier detection using IQR
outlier_info = {}
for column in df.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_info[column] = {
        "Number of Outliers": outliers.shape[0],
        "Percent of Outliers": 100 * outliers.shape[0] / df.shape[0]
    }

outlier_df = pd.DataFrame(outlier_info).T

# Print results
print("=== Summary Statistics ===")
print(summary_stats)

print("\n=== Outlier Summary ===")
print(outlier_df.sort_values(by="Percent of Outliers", ascending=False))

# Visualizations
sns.set(style="whitegrid", palette="muted")

# Histograms
df.hist(bins=30, figsize=(15, 12), layout=(4, 3), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of All Features", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

# Boxplots
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns):
    plt.subplot(4, 3, i+1)
    sns.boxplot(x=df[column], color='lightcoral')
    plt.title(column)
plt.suptitle("Boxplots of All Features", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
