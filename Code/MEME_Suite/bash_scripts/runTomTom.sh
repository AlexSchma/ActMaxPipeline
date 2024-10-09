#!/bin/bash

# Define the general input and output directories
directory_name="Jordi_ActMax"
input_dir="Data/Results/${directory_name}/MEME_Results"
output_dir="Data/Results/${directory_name}/Tomtom_Results"

# Define the ARABIDOPSIS DNA database path (adjust this path to where your ARABIDOPSIS database is located)
database_path1="Data/ARABD/ArabidopsisDAPv1.meme"
database_path2="Data/ARABD/ArabidopsisPBM_20140210.meme"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop over all directories in the input directory that end with 'meme_output'
for meme_dir in "$input_dir"/*meme_output; do
    if [ -d "$meme_dir" ]; then
        # Extract the base directory name (e.g., 'ABA_over' from 'ABA_over_meme_output')
        base_name=$(basename "$meme_dir" _meme_output)
        
        # Define the corresponding output directory (e.g., 'ABA_over_tomtom_output')
        tomtom_output_dir="$output_dir/${base_name}_tomtom_output"
        
        # Create the output directory for TomTom results
        #mkdir -p "$tomtom_output_dir"
        
        # Look for the meme.html file in the current meme_output directory
        meme_file="$meme_dir/meme.html"
        
        # Check if the meme.html file exists before running TomTom
        if [ -f "$meme_file" ]; then
            tomtom -o "$tomtom_output_dir" -eps -png -thresh 0.05 -dist ed "$meme_file" "$database_path1" "$database_path2"
            echo "TomTom search completed for $meme_file, results saved in $tomtom_output_dir"
        else
            echo "No meme.html file found in $meme_dir, skipping..."
        fi
    fi
done

echo "All TomTom searches are completed!"
