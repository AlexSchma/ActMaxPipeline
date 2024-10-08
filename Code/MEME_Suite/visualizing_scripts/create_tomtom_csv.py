import os
import re
import pandas as pd

def parse_tomtom_tsv(filepath, condition):
    """Extract p-value, E-value, and q-value for each motif from the Tomtom TSV file."""
    df = pd.read_csv(filepath, sep='\t', comment='#')
    df = df.sort_values(by="q-value")
    df['Condition'] = condition
    df['TF'] = df['Target_ID'].apply(lambda x: x.split('.')[1].split('_')[0] if len(x.split('.')) > 1 else x)
    df['E-value'] = df['E-value'].apply(lambda x: f"{x:.2e}")
    df['q-value'] = df['q-value'].apply(lambda x: f"{x:.2e}")
    df['p-value'] = df['p-value'].apply(lambda x: f"{x:.2e}")
    return df[['Condition', 'Query_ID', 'TF', 'Target_ID', 'E-value', 'q-value', 'p-value']]

def process_directory(base_dir):
    directory = 'tomtom_results_DO8_108_endmodel.pth_euc_q_value'
    path = os.path.join(base_dir, directory)
    condition_directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.endswith('_tomtom_output')]
    num_conditions = len(condition_directories)
    
    df_list = []
    for condition_dir in sorted(condition_directories):
        condition = condition_dir.split('_tomtom_output')[0]
        condition_path = os.path.join(path, condition_dir)
        tsv_file = os.path.join(condition_path, "tomtom.tsv")
        if not os.path.exists(tsv_file):
            continue
        df = parse_tomtom_tsv(tsv_file, condition)
        if df.empty:
            continue
        df_list.append(df)
    
    if df_list:
        result_df = pd.concat(df_list)
        result_df = result_df.reset_index(drop=True)
        csv_filename = os.path.join(base_dir, f"{directory}.csv")
        result_df.to_csv(csv_filename, index=True)
        print(f"CSV generated: {csv_filename}")
    else:
        print("No motifs found.")


# Base directory where your data is stored
base_dir = ''  # Adjust as needed.
process_directory(base_dir)
