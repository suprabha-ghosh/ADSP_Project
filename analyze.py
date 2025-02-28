import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import os

# Create output directory
output_dir = "mushra_test_analysis"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load MUSHRA results
mushra_results = pd.read_csv("csv/mushra_test_results.csv")

# Print the data structure
print("Data Types:")
print(mushra_results.dtypes)
print("\nFirst few rows:")
print(mushra_results.head())

# Separate participant info and ratings
rating_columns = ['Personalised HRTF Rating', 'Benchmark HRTF Rating']

# Calculate statistics for ratings only
means = mushra_results[rating_columns].mean()
stds = mushra_results[rating_columns].std()

print("\nMean Ratings:")
print(means)
print("\nStandard Deviations:")
print(stds)

# Create bar plot
plt.figure(figsize=(10, 6))
x = np.arange(len(rating_columns))
plt.bar(x, means, yerr=stds, capsize=5, width=0.4)

# Customize plot
plt.xlabel("HRTF Type")
plt.ylabel("MUSHRA Score")
plt.title("MUSHRA Test Results: Personalised vs Benchmark HRTFs")
plt.xticks(x, ['Personalised HRTF', 'Benchmark HRTF'])
plt.ylim(0, 100)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add value labels on bars
for i, v in enumerate(means):
    plt.text(i, v + 1, f'{v:.1f}', ha='center')

plt.tight_layout()
# Save bar plot
bar_plot_file = os.path.join(output_dir, 'bar_plot.png')
plt.savefig(bar_plot_file)
plt.close()

# Statistical Analysis (Paired t-test)
try:
    t_stat, p_value = ttest_rel(
        mushra_results['Personalised HRTF Rating'],
        mushra_results['Benchmark HRTF Rating']
    )
    print("\nStatistical Analysis:")
    print(f"Paired t-test results:")
    print(f"t-statistic = {t_stat:.2f}")
    print(f"p-value = {p_value:.4f}")

    if p_value < 0.05:
        print("✅ Significant difference found between Personalised and Benchmark HRTFs (p < 0.05)")
    else:
        print("❌ No significant difference found (p >= 0.05)")
except Exception as e:
    print(f"\nError performing t-test: {str(e)}")

# Detailed summary statistics
print("\nDetailed Summary Statistics:")
summary_stats = mushra_results[rating_columns].describe()
print(summary_stats)

# Individual participant results
print("\nIndividual Participant Results:")
participant_results = pd.DataFrame({
    'Participant': mushra_results['Participant'],
    'Personalised HRTF': mushra_results['Personalised HRTF Rating'],
    'Benchmark HRTF': mushra_results['Benchmark HRTF Rating'],
    'Difference': mushra_results['Personalised HRTF Rating'] - mushra_results['Benchmark HRTF Rating']
})
print(participant_results)

# Save results to CSV
try:
    summary_file = os.path.join(output_dir, 'mushra_analysis_results.csv')
    with open(summary_file, 'w') as f:
        f.write("MUSHRA Test Analysis Results\n\n")
        f.write("Mean Ratings:\n")
        means.to_csv(f)
        f.write("\nStandard Deviations:\n")
        stds.to_csv(f)
        f.write("\nStatistical Analysis:\n")
        f.write(f"t-statistic,{t_stat:.4f}\n")
        f.write(f"p-value,{p_value:.4f}\n")
        f.write("\nDetailed Summary Statistics:\n")
        summary_stats.to_csv(f)
        f.write("\nIndividual Results:\n")
        participant_results.to_csv(f)
    print(f"\nAnalysis results saved to {summary_file}")
except Exception as e:
    print(f"\nError saving results: {str(e)}")

# Additional visualizations
plt.figure(figsize=(12, 6))

# Box plot
plt.subplot(1, 2, 1)
sns.boxplot(data=mushra_results[rating_columns])
plt.title("Distribution of Ratings")
plt.ylabel("MUSHRA Score")
plt.xticks([0, 1], ['Personalised HRTF', 'Benchmark HRTF'])

# Individual participant plot
plt.subplot(1, 2, 2)
for i in range(len(mushra_results)):
    plt.plot([0, 1], 
             [mushra_results['Personalised HRTF Rating'].iloc[i], 
              mushra_results['Benchmark HRTF Rating'].iloc[i]], 
             'o-', alpha=0.3)
plt.xticks([0, 1], ['Personalised HRTF', 'Benchmark HRTF'])
plt.title("Individual Participant Ratings")
plt.ylabel("MUSHRA Score")
plt.ylim(0, 100)

plt.tight_layout()
# Save additional plots
additional_plots_file = os.path.join(output_dir, 'additional_plots.png')
plt.savefig(additional_plots_file)
plt.close()

# Save participant results separately
participant_results_file = os.path.join(output_dir, 'participant_results.csv')
participant_results.to_csv(participant_results_file, index=False)

print("\nFiles saved in the 'mushra_test_analysis' folder:")
print(f"1. Bar Plot: bar_plot.png")
print(f"2. Additional Plots: additional_plots.png")
print(f"3. Analysis Results: mushra_analysis_results.csv")
print(f"4. Participant Results: participant_results.csv")

# Create a summary text file
summary_text_file = os.path.join(output_dir, 'analysis_summary.txt')
with open(summary_text_file, 'w', encoding='utf-8') as f:
    f.write("MUSHRA Test Analysis Summary\n")
    f.write("==========================\n\n")
    f.write("Mean Ratings:\n")
    f.write(str(means))
    f.write("\n\nStandard Deviations:\n")
    f.write(str(stds))
    f.write("\n\nStatistical Analysis:\n")
    f.write(f"t-statistic = {t_stat:.2f}\n")
    f.write(f"p-value = {p_value:.4f}\n")
    f.write("\nConclusion: ")
    if p_value < 0.05:
        f.write("Significant difference found between Personalised and Benchmark HRTFs (p < 0.05)")
    else:
        f.write("No significant difference found (p >= 0.05)")

print(f"5. Analysis Summary: analysis_summary.txt")
