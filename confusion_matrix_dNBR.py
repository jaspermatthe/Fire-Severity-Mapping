import matplotlib.pyplot as plt
import numpy as np

# Create an empty 4x4 confusion matrix
confusion_matrix = [
    [95, 7, 1, 0],
    [8, 76, 26, 1],
    [2, 19, 20, 11],
    [5, 1, 7, 46]
]

class_names = ['Unchanged', 'Low', 'Moderate', 'High']

# Calculate producer and user accuracies
producer_accuracies = [confusion_matrix[i][i] / sum(confusion_matrix[i]) for i in range(len(class_names))]
user_accuracies = [confusion_matrix[i][i] / sum(confusion_matrix[j][i] for j in range(len(class_names))) for i in range(len(class_names))]

# Calculate overall accuracy
overall_accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

# Calculate kappa coefficient
total = np.sum(confusion_matrix)
po = np.trace(confusion_matrix) / total
pe = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / (total * total)
kappa = (po - pe) / (1 - pe)

# Define class labels
class_names_x = [f'{class_name}, \n UA: {int(user_accuracies[i] * 100)}%' for i, class_name in enumerate(class_names)]
class_names_y = [f'{class_name}, \n PA: {int(producer_accuracies[i] * 100)}%' for i, class_name in enumerate(class_names)]

# Create the figure and axes
fig, ax = plt.subplots()

# Plot the confusion matrix
cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

# Set labels for the x and y axes with class names
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names_x)
ax.set_yticklabels(class_names_y)
# Move x-axis tick labels to the top
ax.xaxis.tick_bottom()

# Move y-axis tick labels to the right
ax.yaxis.tick_left()

# Add colorbar
cbar = fig.colorbar(cax)

# Add text annotations to the cells
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(i, j, confusion_matrix[j][i], ha='center', va='center', color='black')

# Add the overall accuracy and kappa coefficient to the title
plt.title(f'Overall Accuracy: {overall_accuracy * 100:.2f}%, Kappa Coefficient: {kappa:.2f}')

# Add x-axis and y-axis titles
ax.set_xlabel('Actual (CBI)')
ax.set_ylabel('Estimated (dNBR)')

# Save the plot with axis titles
plt.savefig('confusion_matrix_dNBR.png', bbox_inches='tight', pad_inches=0.1)

plt.show()
