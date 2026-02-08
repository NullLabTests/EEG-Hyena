import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for headless
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Boxes
ax.add_patch(Rectangle((0.5, 2), 2, 1, fill=None, edgecolor='black'))
ax.text(1.5, 2.5, 'EEG Input', ha='center', va='center')
ax.add_patch(Rectangle((3, 2), 2, 1, fill=None, edgecolor='black'))
ax.text(4, 2.5, 'Preprocess\n(Filter, Epoch, PCA)', ha='center', va='center')
ax.add_patch(Rectangle((5.5, 2), 2, 1, fill=None, edgecolor='black'))
ax.text(6.5, 2.5, 'Hyena Model\n(Conv Layers, EWC)', ha='center', va='center')
ax.add_patch(Rectangle((8, 2), 2, 1, fill=None, edgecolor='black'))
ax.text(9, 2.5, 'Text Output', ha='center', va='center')

# Arrows
ax.add_patch(FancyArrowPatch((2.5, 2.5), (3, 2.5), arrowstyle='->', mutation_scale=20))
ax.add_patch(FancyArrowPatch((5, 2.5), (5.5, 2.5), arrowstyle='->', mutation_scale=20))
ax.add_patch(FancyArrowPatch((7.5, 2.5), (8, 2.5), arrowstyle='->', mutation_scale=20))

plt.savefig('arch.png', bbox_inches='tight')
