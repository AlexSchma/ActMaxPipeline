import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re

def parse_meme_text(filepath):
    """Extract E-values for each motif from the MEME text file."""
    with open(filepath, 'r') as file:
        content = file.read()
    e_values = re.findall(r'MOTIF (\S+) MEME-\d+.*?E-value = ([\de\.-]+)', content, re.DOTALL)
    return [e_value for motif, e_value in e_values]

def add_motif_images_to_pdf(motif_paths, e_values, condition, axes, idx):
    """Add motif images and their E-values to the subplot axes, including condition labels."""
    for i, motif_path in enumerate(sorted(motif_paths)):
        ax = axes[idx, i]
        img = Image.open(motif_path)
        ax.imshow(img)
        ax.axis('off')
        motif_name = os.path.basename(motif_path).split('.')[0]
        ax.set_title(f"E-value: {e_values[i]}", fontsize=8)
    
    axes[idx, 0].text(-0.2, 0.5, condition, transform=axes[idx, 0].transAxes,
                       horizontalalignment='right', verticalalignment='center',
                       fontsize=9, rotation=90)

def process_directory(base_dir):
    directory = 'results_DO8_108_endmodel2'
    path = os.path.join(base_dir, directory)
    condition_directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.endswith('_meme_output')]
    num_conditions = len(condition_directories)
    
    pdf_filename = os.path.join(base_dir, f"{directory}_motifs.pdf")
    with PdfPages(pdf_filename) as pdf_pages:
        fig, axes = plt.subplots(num_conditions, 3, figsize=(8, 3 * num_conditions), gridspec_kw={'hspace': 0.5, 'wspace': 0})
        for i, condition_dir in enumerate(sorted(condition_directories)):
            condition = condition_dir.split('_meme_output')[0]
            condition_path = os.path.join(path, condition_dir)
            txt_file = next(f for f in os.listdir(condition_path) if f.endswith('.txt'))
            e_values = parse_meme_text(os.path.join(condition_path, txt_file))
            motif_images = [os.path.join(condition_path, file) for file in os.listdir(condition_path) if file.endswith('.png') and 'rc' not in file]
            add_motif_images_to_pdf(motif_images, e_values, condition, axes, i)
        fig.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.05)
        fig.text(0.1, 0.98, 'Condition', fontsize=10, weight='bold', ha='center', va='top')
        pdf_pages.savefig(fig)
        plt.close(fig)

# Base directory where your data is stored
base_dir = ''  # Adjust as needed.
process_directory(base_dir)
