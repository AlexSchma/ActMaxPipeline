#!/bin/bash

directory_name="Jordi_ActMax"

# Directory containing your FASTA files
fasta_directory="Data/Results/${directory_name}"

# Output directory for MEME results
output_directory="Data/Results/${directory_name}/MEME_Results"

# Create output directory if it doesn't exist
mkdir -p "${output_directory}"

# Loop through each FASTA file in the directory
for fasta_file in "${fasta_directory}"/*.fasta
do
    echo "Processing file: $fasta_file"
    # Extract the filename without the directory and extension for naming the output
    base_name=$(basename "$fasta_file" .fasta)

    # Run MEME command with the 'anr' model for any number of repetitions
    meme "$fasta_file" -dna -mod anr -nmotifs 3 -minw 6 -maxw 8 -o "${output_directory}/${base_name}_meme_output"
    echo "Completed: ${output_directory}/${base_name}_meme_output"
done
