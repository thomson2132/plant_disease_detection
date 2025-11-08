import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# If using a CSV file with labels
# df = pd.read_csv('your_dataset.csv')
# Replace 'label_column' with your actual label column name
# class_counts = df['label_column'].value_counts()

# If your dataset is structured in folders (Image Classification)
import os
from collections import Counter

dataset_path = "C:/Users/thoms/OneDrive/Desktop/Capstone/PlantVillage"  # Replace with your dataset path
class_names = os.listdir(dataset_path)

class_counts = {}
for class_name in class_names:
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        count = len(os.listdir(class_dir))
        class_counts[class_name] = count

# Convert to a Pandas series for analysis
class_counts_series = pd.Series(class_counts).sort_values(ascending=False)

# Print counts
print("Class distribution:\n", class_counts_series)

# Plot the class distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=class_counts_series.index, y=class_counts_series.values)
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.ylabel("Number of Samples")
plt.xlabel("Class Labels")
plt.tight_layout()
plt.show()

# Check for imbalance
max_count = class_counts_series.max()
min_count = class_counts_series.min()
imbalance_ratio = max_count / min_count

if imbalance_ratio > 1.5:  # Common threshold
    print(f"\n⚠️ Potential imbalance detected! Max/Min ratio: {imbalance_ratio:.2f}")
else:
    print("\n✅ Classes appear balanced.")
