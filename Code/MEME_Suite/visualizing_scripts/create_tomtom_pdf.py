import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re
import pandas as pd

def parse_tomtom_tsv(filepath):
    """Extract p-value, E-value, and q-value for each motif from the Tomtom TSV file."""
    df = pd.read_csv(filepath, sep='\t', comment='#')
    df = df.sort_values(by="q-value").head(9)
    return df

def add_motif_images_to_pdf(motif_paths, df, condition, axes, row_idx,motif_counter):
    """Add motif images and their p/e/q-values to the subplot axes, including condition labels."""
    num_motifs = len(motif_paths)
    for i in range(3):  # Loop through 3 columns
        ax = axes[row_idx, i]
        if i < num_motifs:
            img = Image.open(motif_paths[i])
            img = img.convert('RGB')  # Convert to RGB mode if it has transparency
            img.thumbnail((600, 400), resample=Image.LANCZOS)
            width, height = img.size
            padded_img = Image.new('RGBA', (600, 400), (255, 255, 255, 0))  # Use RGBA for transparency
            padded_img.paste(img, ((600 - width) // 2, (400 - height) // 2))
            ax.imshow(padded_img)
            ax.axis('off')
            row = df.iloc[i]
            ax.set_title(f"{str(i+1+motif_counter)})\n{row['Target_ID']}\n{row['p-value']:3.2e}\n{row['E-value']:3.2e}\n{row['q-value']:3.2e}", fontsize=7, ha='left',loc="left",x=0.1)
        else:
            ax.axis('off')  # Hide unused subplots

    # Add condition label on the first column's left side
    axes[row_idx, 0].text(-0.1, 0.5, condition, transform=axes[row_idx, 0].transAxes,
                          horizontalalignment='right', verticalalignment='center',
                          fontsize=9)
    axes[row_idx, 0].text(-200,-25,"Nr:\nTarget ID:\np-value:\nE-value:\nq-value:", fontsize=7)

def process_directory(base_dir):
    directory = 'tomtom_results_DO8_108_endmodel.pth_euc_q_value'
    path = os.path.join(base_dir, directory)
    condition_directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.endswith('_tomtom_output')]
    num_conditions = len(condition_directories)
    
    pdf_filename = os.path.join(base_dir, f"{directory}_motifs_summary.pdf")
    motif_counter = 0
    with PdfPages(pdf_filename) as pdf_pages:
        for condition_dir in sorted(condition_directories):
            condition = condition_dir.split('_tomtom_output')[0]
            condition_path = os.path.join(path, condition_dir)
            tsv_file = os.path.join(condition_path, "tomtom.tsv")
            if not os.path.exists(tsv_file):
                continue
            df = parse_tomtom_tsv(tsv_file)
            if df.empty:
                continue

            motif_images = []
            for row in df.itertuples():
                query_id = row.Query_ID
                target_id = row.Target_ID
                orientation = row.Orientation

                image_name_0 = f"align_{query_id}_0_{orientation}{target_id}.png"
                image_name_1 = f"align_{query_id}_1_{orientation}{target_id}.png"

                image_path = os.path.join(condition_path, image_name_0)
                if not os.path.exists(image_path):
                    image_path = os.path.join(condition_path, image_name_1)
                    if not os.path.exists(image_path):
                        continue

                motif_images.append(image_path)

            num_rows = (len(motif_images) + 2) // 3  # Number of rows required (3 images per row)
            fig, axes = plt.subplots(num_rows, 3, figsize=(8, 3 * num_rows), gridspec_kw={'hspace': 0.5, 'wspace': 0})
            
            # If there is only one row, axes may not be a 2D array, we need to handle that
            if num_rows == 1:
                axes = axes.reshape(1, 3)

            for row_idx in range(num_rows):
                start_idx = row_idx * 3
                add_motif_images_to_pdf(motif_images[start_idx:start_idx + 3], df.iloc[start_idx:start_idx + 3], condition, axes, row_idx, motif_counter)
                motif_counter += 3
            
            fig.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.05)
            pdf_pages.savefig(fig)
            plt.close(fig)
            motif_counter = 0
            

    print(f"PDF generated: {pdf_filename}")

# Base directory where your data is stored
base_dir = ''  # Adjust as needed.
process_directory(base_dir)
