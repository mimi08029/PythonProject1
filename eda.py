import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_dataset import make_datasets, make_pairs
from load_osu_map import load_osu_maps
def depack(data_loader):
    maps, all_maps = [], []
    for batch in data_loader:
        pass

data_list = load_osu_maps()
data = make_datasets(depack(make_pairs(data_list, 5)))

df = pd.DataFrame(data)

print("Shape:", df.shape)
print("Columns:", df.columns)
print("\nHead:")
print(df.head())

print("\nDescribe:")
print(df.describe(include="all"))

print("\nMissing values per column:")
print(df.isnull().sum())

for col in df.select_dtypes(include=["int64", "float64"]).columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

for col in df.select_dtypes(include=["object"]).columns:
    plt.figure(figsize=(6,4))
    df[col].value_counts().head(20).plot(kind="bar")
    plt.title(f"Top categories in {col}")
    plt.show()
