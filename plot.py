import pandas as pd
import matplotlib.pyplot as plt

# === Config ===
csv_file = 'accuracy_to_parameters_cifar10.csv'      # Replace with your actual CSV path
x_col = 'Parameters'               # Replace with your actual x-axis column name
y_col = 'Accuracies'               # Replace with your actual y-axis column name
output_file = 'plot_cifar10.png'            # Name of the output PNG file

# === Read CSV ===
df = pd.read_csv(csv_file)

# === Plot ===
plt.figure(figsize=(8, 5))
plt.plot(df[x_col], df[y_col], marker='o')  # Use scatter() if you prefer dots only
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title(f'{y_col} vs {x_col}')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Save to PNG ===
plt.savefig(output_file, dpi=300)   # You can increase dpi for higher resolution
plt.show()
