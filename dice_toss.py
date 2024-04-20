import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats

# Set up the figure and axes
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Initialize variables
num_trials = 1000
dice_rolls = 7
dice_faces = 6
means = []
p_values = []

# Generate original distribution
original_distribution = np.random.randint(1, dice_faces + 1, size=num_trials * dice_rolls)
original_distribution_means = np.mean(original_distribution.reshape(-1, dice_rolls), axis=1)

# Histogram of means
hist_ax = axs[0, 0]
hist_ax.set_title('Histogram of Means')
hist_ax.set_xlabel('Mean')
hist_ax.set_ylabel('Frequency')

# QQ plot
qq_ax = axs[0, 1]
qq_ax.set_title('QQ Plot')
qq_ax.set_xlabel('Theoretical Quantiles')
qq_ax.set_ylabel('Sample Quantiles')

# Original distribution
orig_dist_ax = axs[1, 0]
orig_dist_ax.set_title('Original Distribution')
orig_dist_ax.set_xlabel('Value')
orig_dist_ax.set_ylabel('Frequency')

# P-values from Shapiro-Wilk Test
p_values_ax = axs[1, 1]
p_values_ax.set_title('Shapiro-Wilk Test p-values')
p_values_ax.set_xlabel('Trial')
p_values_ax.set_ylabel('p-value')

# Dice toss distribution
dice_dist_ax = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
dice_dist_ax.set_title('Dice Toss Distribution')
dice_dist_ax.set_xlabel('Value')
dice_dist_ax.set_ylabel('Frequency')

# Hide unused subplot
shapiro_ax = axs[1, 2]
shapiro_ax.axis('off')

# Function to update plots
def update(frame):
    dice_toss = np.random.randint(1, dice_faces + 1, size=dice_rolls)
    means.append(np.mean(dice_toss))

    # Update histogram of means
    hist_ax.clear()
    hist_ax.hist(means, bins=30, alpha=0.75)
    hist_ax.set_title('Histogram of Means')
    hist_ax.set_xlabel('Mean')
    hist_ax.set_ylabel('Frequency')
    hist_ax.text(0.05, 0.9, f'Trials: {frame}', transform=hist_ax.transAxes)

    # Update QQ plot
    qq_ax.clear()
    stats.probplot(means, dist="norm", plot=qq_ax)
    qq_ax.set_title('QQ Plot')
    qq_ax.set_xlabel('Theoretical Quantiles')
    qq_ax.set_ylabel('Sample Quantiles')

    # Update dice toss distribution plot
    dice_dist_ax.clear()
    dice_dist_ax.hist(dice_toss, bins=range(1, dice_faces + 2), alpha=0.75, align='left')
    dice_dist_ax.set_title('Dice Toss Distribution')
    dice_dist_ax.set_xlabel('Value')
    dice_dist_ax.set_ylabel('Frequency')

    # Update original distribution plot
    orig_dist_ax.clear()
    orig_dist_ax.hist(original_distribution_means[:frame], bins=30, alpha=0.75)
    orig_dist_ax.set_title('Original Distribution')
    orig_dist_ax.set_xlabel('Value')
    orig_dist_ax.set_ylabel('Frequency')

    # Update p-values plot
    shapiro_statistic, p_value = stats.shapiro(dice_toss)
    p_values.append(p_value)

    p_values_ax.clear()
    p_values_ax.plot(range(len(p_values)), p_values, marker='o', linestyle='-')
    p_values_ax.set_title('Shapiro-Wilk Test p-values')
    p_values_ax.set_xlabel('Trial')
    p_values_ax.set_ylabel('p-value')
    p_values_ax.set_ylim(0, 1)


# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_trials, interval=500)

plt.tight_layout()
plt.show()
