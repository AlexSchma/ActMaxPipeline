import random
import sys
sys.path.append("./")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import chi2_contingency
#from Bio.Seq import Seq
import pandas as pd
import os
import seaborn as sns
def percent_with_motifs(nuc_seqs,motifs):
    found = 0
    for nuc_seq in nuc_seqs:
        if (motifs[0] in nuc_seq or motifs[1] in nuc_seq or motifs[2] in nuc_seq):
            found += 1 
    return found/len(nuc_seqs)
def generate_random_sequence(length=100):
    bases = "ACGT"
    return ''.join(random.choice(bases) for _ in range(length))

def find_motif_positions(dna, motifs):
    for motif in motifs:
        pos = dna.find(motif)
        while pos != -1:
            print(f"Motif '{motif}' found at position {pos}")
            pos = dna.find(motif, pos + 1)

# Function to count motif occurrences and co-occurrences in a list of sequences
def count_motifs(sequences, motifs):
    motif_counts = {motif: 0 for motif in motifs}
    co_occurrence_counts = {(motif1, motif2): 0 for motif1 in motifs for motif2 in motifs if motif1 != motif2}

    for seq in sequences:
        found_motifs = []

        for motif in motifs:
            count = seq.count(motif)
            if count > 0:
                motif_counts[motif] += count
                found_motifs.append(motif)

        for i in range(len(found_motifs)):
            for j in range(i + 1, len(found_motifs)):
                co_occurrence_counts[(found_motifs[i], found_motifs[j])] += 1
                co_occurrence_counts[(found_motifs[j], found_motifs[i])] += 1

    return motif_counts, co_occurrence_counts
def count_motifs_and_plot(sequences, motifs):
    motif_counts_per_sequence = {motif: [] for motif in motifs}

    for seq in sequences:
        for motif in motifs:
            count = seq.count(motif)
            motif_counts_per_sequence[motif].append(count)

    # Plotting histograms for each motif
    for motif, counts in motif_counts_per_sequence.items():
        plt.figure(figsize=(10, 6))
        plt.hist(counts, bins=range(min(counts), max(counts) + 2), align='left', color='skyblue', edgecolor='black')
        plt.title(f"Histogram of {motif} occurrences per sequence")
        plt.xlabel('Occurrences')
        plt.ylabel('Number of Sequences')
        plt.xticks(range(min(counts), max(counts) + 1))
        plt.grid(axis='y', alpha=0.75)
        plt.show()

def plot_motif_occurrences(results, motifs, results_dir):
    for model_name, conditions_data in results.items():
        df = pd.DataFrame.from_dict(conditions_data, orient='index', columns=motifs)
        df.plot(kind='bar', figsize=(10, 6))
        plt.title(f'Average Motif Occurrences per Sequence for {model_name}')
        plt.ylabel('Average Occurrences')
        plt.xlabel('Condition')
        plt.xticks(rotation=0)
        plt.legend(title='Motif')
        plot_filename = f'{model_name}_motif_occurrences.png'
        plt.savefig(os.path.join(results_dir, plot_filename))
        plt.close()

def plot_motif_occurrences(results, motifs, results_dir, include_stds=False, relaxation_param=0):
    for model_name, conditions_data in results.items():
        # Creating DataFrames for average and standard deviation data
        avg_data = {condition: data['avg'] for condition, data in conditions_data.items()}
        df_avg = pd.DataFrame.from_dict(avg_data, orient='index', columns=motifs)

        if include_stds:
            std_data = {condition: data['std'] for condition, data in conditions_data.items()}
            df_std = pd.DataFrame.from_dict(std_data, orient='index', columns=motifs)
            ax = df_avg.plot(kind='bar', yerr=df_std, capsize=4, figsize=(10, 6), legend=True)
        else:
            ax = df_avg.plot(kind='bar', figsize=(10, 6), legend=True)
        
        # Adjusting the title to include the relaxation parameter
        if relaxation_param > 0:
            plt.title(f'Average Motif Occurrences per Sequence for {model_name} (Relaxed by {relaxation_param} positions)')
        else:
            plt.title(f'Average Motif Occurrences per Sequence for {model_name}')
        
        plt.ylabel('Average Occurrences')
        plt.xlabel('Condition')
        plt.xticks(rotation=0)
        plt.legend(title='Motif')

        # Constructing the filename to include information about standard deviations
        std_suffix = '_with_stds' if include_stds else ''
        relaxation_suffix = f"_relaxed_by_{relaxation_param}_" if relaxation_param>0 else "_"
        plot_filename = f'{model_name}_motif_occurrences{relaxation_suffix}{std_suffix}.png'
        plt.savefig(os.path.join(results_dir, plot_filename))
        plt.close()

def plot_roc_curves(Base_model_name,roc_data,results_dir):
    plt.figure()
    for model_name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plot_filename = f'{Base_model_name}_roc_curves.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()

def plot_accuracy_bar(Base_model_name,accuracies,results_dir):
    plt.figure()
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plot_filename = f'{Base_model_name}_accuracy_bar.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()


def calculate_relative_increases(results, motifs):
    relative_increases = {motif: [] for motif in motifs}  # Store relative increases for each motif

    for _, conditions_data in results.items():
        for cond, motif_data in conditions_data.items():
            # Convert the condition to an integer for direct comparison
            cond = int(float(cond))
            # For each motif, calculate the relative increase in its corresponding condition
            for motif_index, motif in enumerate(motifs):
                if motif_index == cond:  # Matching condition and motif
                    # Average occurrence of the motif under other conditions
                    avg_other_conditions = np.mean([v[motif] for k, v in conditions_data.items() if int(float(k)) != cond])
                    # Relative increase = occurrence under matching condition - average occurrence under other conditions
                    relative_increase = motif_data[motif] - avg_other_conditions
                    relative_increases[motif].append(relative_increase)

    # Aggregate the relative increases across all models
    aggregated_increases = {motif: np.mean(increases) for motif, increases in relative_increases.items()}
    return aggregated_increases

def plot_relative_increases(aggregated_increases, results_dir,Base_model_name):
    # Convert the aggregated increases to a DataFrame for easy plotting
    df = pd.DataFrame(list(aggregated_increases.items()), columns=['Motif', 'Relative Increase'])

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Motif', y='Relative Increase', data=df)
    plt.title('Relative Increase in Motif Occurrences Under Matching Conditions')
    plt.ylabel('Average Relative Increase')
    plt.xlabel('Motif')

    # Saving the plot
    plot_filename = f'{Base_model_name}_relative_increase_in_motif_occurrences.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()

def average_motif_occurrences_uncond_relaxed(motifs, nuc_seqs_per_model, relaxation_param=2):
    avg_occurrences = pd.DataFrame(index=motifs)
    stds = pd.DataFrame(index=motifs)
    relaxed_motifs_to_original_motif = {}
    new_motifs_len = len(motifs[0]) - relaxation_param
    
    for motif in motifs:
        for step in range(relaxation_param + 1):
            shortened_motif = motif[step:step + new_motifs_len]
            if shortened_motif not in relaxed_motifs_to_original_motif:
                relaxed_motifs_to_original_motif[shortened_motif] = motif
            else:
                del relaxed_motifs_to_original_motif[shortened_motif]
                print(f"Duplicate motif found and removed from analysis: {shortened_motif}")
                
    for model_name, nuc_seqs in nuc_seqs_per_model.items():
        occurrences_per_seq = {motif: [] for motif in motifs}
        
        for seq in nuc_seqs:
            counts = {motif: 0 for motif in motifs}
            for motif in relaxed_motifs_to_original_motif.keys():
                counts[relaxed_motifs_to_original_motif[motif]] += seq.count(motif)
            for motif, count in counts.items():
                occurrences_per_seq[motif].append(count)
                
        # Calculate averages and standard deviations
        for motif in motifs:
            occurrences = occurrences_per_seq[motif]
            avg_occurrences.loc[motif, model_name] = sum(occurrences) / len(occurrences)
            stds.loc[motif, model_name] = pd.Series(occurrences).std()

    return avg_occurrences, stds


def plot_motif_occurrences_uncond(avg_occurrences, stds, results_dir, Basemodel_name, include_stds=True,relaxation_param=2):
    # Check if standard deviations should be included in the plot
    yerr = stds if include_stds else None

    # Plotting the average occurrences with or without error bars for standard deviations
    avg_occurrences.plot(kind='bar', figsize=(10, 6), yerr=yerr, capsize=4)
    if relaxation_param == 0:
        plt.title('Average Motif Occurrences per Sequence per Model')
    else:
        new_len = len(avg_occurrences.index[0]) - relaxation_param
        plt.title(f'Average Relaxed Motif ({new_len}mers) Occurrences per Sequence per Model')
    plt.ylabel('Average Occurrences')
    plt.xlabel('Motif')
    plt.xticks(rotation=45)
    plt.legend(title='Model')

    # Adjusting the filename based on whether stds are included
    plot_filename_suffix = '_with_stds' if include_stds else ''
    relaxation_suffix = f"_relaxed_by_{relaxation_param}_" if relaxation_param > 0 else ''
    plot_filename = f'{Basemodel_name}_motif_occurrences{relaxation_suffix}{plot_filename_suffix}.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    
    plt.close()

def average_motif_occurrences_relaxed_with_std(motifs, nuc_seqs_per_model, relaxation_param=2):
    results = {}
    relaxed_motifs_to_original_motif = {}
    new_motifs_len = len(motifs[0]) - relaxation_param

    # Relaxing motifs
    for motif in motifs:
        for step in range(relaxation_param + 1):
            shortened_motif = motif[step:step + new_motifs_len]
            if shortened_motif not in relaxed_motifs_to_original_motif:
                relaxed_motifs_to_original_motif[shortened_motif] = motif
            else:
                del relaxed_motifs_to_original_motif[shortened_motif]
                print(f"Duplicate motif found and removed from analysis: {shortened_motif}")

    # Calculating average occurrences and standard deviations
    for model_name, conditions_data in nuc_seqs_per_model.items():
        for condition, nuc_seqs in conditions_data.items():
            occurrences_per_seq = {motif: [] for motif in motifs}
            
            for seq in nuc_seqs:
                counts = {motif: 0 for motif in motifs}
                for relaxed_motif, original_motif in relaxed_motifs_to_original_motif.items():
                    counts[original_motif] += seq.count(relaxed_motif)
                
                for motif, count in counts.items():
                    occurrences_per_seq[motif].append(count)
            
            avg_counts = {motif: sum(occurrences) / len(occurrences) for motif, occurrences in occurrences_per_seq.items()}
            std_counts = {motif: pd.Series(occurrences).std() for motif, occurrences in occurrences_per_seq.items()}
            
            results.setdefault(model_name, {}).setdefault(condition, {})['avg'] = avg_counts
            results[model_name][condition]['std'] = std_counts

    return results



import numpy as np

def aggregate_motif_occurrences(results, motifs):
    # Initialize dictionaries to hold the aggregated averages and standard deviations
    aggregated_avgs = {condition: {motif: [] for motif in motifs} for condition in results[next(iter(results))]}
    aggregated_stds = {condition: {motif: [] for motif in motifs} for condition in results[next(iter(results))]}

    # Aggregate the average occurrences and standard deviations data
    for _, conditions_data in results.items():
        for condition, data in conditions_data.items():
            avg_data = data['avg']
            std_data = data['std']
            for motif in motifs:
                aggregated_avgs[condition][motif].append(avg_data[motif])
                aggregated_stds[condition][motif].append(std_data[motif])

    # Calculate the mean and standard deviation of the aggregated data
    aggregated_results = {}
    for condition in aggregated_avgs:
        aggregated_results[condition] = {}
        for motif in motifs:
            mean_avg = np.mean(aggregated_avgs[condition][motif])
            # For standard deviations, use pooled standard deviation if appropriate or calculate a new mean
            mean_std = np.mean(aggregated_stds[condition][motif])
            aggregated_results[condition][motif] = {'avg': mean_avg, 'std': mean_std}

    return aggregated_results

def plot_aggregated_motif_occurrences(Base_model_name, aggregated_results,motifs, results_dir, include_std=True,relaxation_param=2):
    # Extracting averages and standard deviations into separate DataFrames
    avg_data = {condition: {motif: aggregated_results[condition][motif]['avg'] for motif in aggregated_results[condition]} for condition in aggregated_results}
    std_data = {condition: {motif: aggregated_results[condition][motif]['std'] for motif in aggregated_results[condition]} for condition in aggregated_results}

    avg_df = pd.DataFrame(avg_data).T  # Transpose to have conditions as rows
    std_df = pd.DataFrame(std_data).T if include_std else None

    # Plotting with or without error bars for standard deviations
    ax = avg_df.plot(kind='bar', yerr=std_df, figsize=(12, 8), capsize=4) if include_std else avg_df.plot(kind='bar', figsize=(12, 8))
    new_motifs_len = len(motifs[0]) - relaxation_param
    if relaxation_param == 0:
        plt.title(f'Average Motif Occurrences per Condition Across All Models' + (' (with Std Dev)' if include_std else ''))
    else:
        plt.title(f'Average Relaxed Motif ({new_motifs_len}mers) Occurrences per Condition Across All Models' + (' (with Std Dev)' if include_std else ''))
    plt.ylabel('Average Occurrence')
    plt.xlabel('Condition')
    plt.xticks(rotation=0)
    plt.legend(title='Motif')
    plt.tight_layout()

    # Saving the plot
    plot_filename_suffix = '_with_stds' if include_std else ''
    relaxation_suffix = f"_relaxed_by_{relaxation_param}_" if relaxation_param > 0 else ''
    plot_filename = f'{Base_model_name}_aggregated_motif_occurrences{relaxation_suffix}{plot_filename_suffix}.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()

def plot_aggregated_preference_scores(aggregated_scores, motifs, results_dir, Base_model_name, relaxation_param=2):
    # Convert aggregated scores to a DataFrame for plotting
    df = pd.DataFrame(aggregated_scores).T  # Transpose to have motifs as rows
    new_motifs_len = len(motifs[0]) - relaxation_param

    # Plotting
    ax = df.plot(kind='bar', figsize=(12, 8), width=0.8)
    if relaxation_param == 0:
        plt.title('Aggregated Preference Scores for Motifs Across All Models')
    else:
        plt.title(f'Aggregated Preference Scores for Relaxed Motifs ({new_motifs_len}mers) Across All Models')
    plt.ylabel('Preference Score')
    plt.xlabel('Motif')
    plt.xticks(rotation=0)
    plt.legend(title='Condition')
    plt.tight_layout()

    # Saving the plot
    plot_filename_suffix = f'_relaxed_{new_motifs_len}mers' if relaxation_param > 0 else ''
    plot_filename = f'{Base_model_name}_aggregated_preference_scores{plot_filename_suffix}.png'
    plt.savefig(os.path.join(results_dir, plot_filename))
    plt.close()
def calculate_preference_scores(results, motifs,use_std=True):
    preference_scores = {motif: {str(cond): [] for cond in range(len(motifs))} for motif in motifs}

    for _, conditions_data in results.items():
        for cond in range(len(motifs)):
            cond_key = str(float(cond)) if str(cond) not in conditions_data else str(cond)
            if use_std:
                total_occurrences = sum(conditions_data[cond_key]['avg'].get(motif, 0) for motif in motifs)
            else:

                total_occurrences = sum(conditions_data[cond_key].get(motif, 0) for motif in motifs)
            for motif in motifs:
                if use_std:
                    motif_occurrence = conditions_data[cond_key]['avg'].get(motif, 0)
                else:

                    motif_occurrence = conditions_data[cond_key].get(motif, 0)
                preference_score = motif_occurrence / total_occurrences if total_occurrences > 0 else 0
                preference_scores[motif][str(cond)].append(preference_score)

    return preference_scores



def aggregate_scores_across_models(preference_scores, motifs):
    aggregated_scores = {motif: {cond: np.mean(scores) if scores else 0 
                                 for cond, scores in cond_scores.items()} 
                         for motif, cond_scores in preference_scores.items()}

    return aggregated_scores