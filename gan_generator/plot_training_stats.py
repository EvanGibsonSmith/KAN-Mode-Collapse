import pandas as pd
import matplotlib.pyplot as plt
import ast

# Load the CSV file
df = pd.read_csv('gan_generator/gan_output_mnist/training_stats.csv')

# Convert the 'Confidence' column from a string to a list of floats
df['Confidence'] = df['Confidence'].apply(lambda x: ast.literal_eval(x))

# Plotting Confidence and Entropy over Epochs
plt.figure(figsize=(10, 6))

# Plot Confidence for each Epoch (average of all confidence values per epoch)
df['Avg Confidence'] = df['Confidence'].apply(lambda x: sum(x) / len(x))

plt.plot(df['Epoch'], df['Avg Confidence'], label='Avg Confidence', color='b', marker='o')

# Plot Entropy
plt.plot(df['Epoch'], df['Entropy'], label='Entropy', color='r', marker='x')

# Add titles and labels
plt.title('Confidence and Entropy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
plt.savefig("out.png")
plt.close()