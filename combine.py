import csv
import os
import glob

# Define the data folder
data_folder = "data"

# Ensure the data folder exists (optional, but safe)
os.makedirs(data_folder, exist_ok=True)

# Output file will be inside the data folder
output_file = os.path.join(data_folder, "combined.csv")

# Open the output CSV for writing
with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
    writer.writerow(["sentence", "language"])   # header

    # Find all CSV files inside the data folder (exclude combined.csv itself)
    for filepath in glob.glob(os.path.join(data_folder, "*.csv")):
        if filepath == output_file:
            continue
        
        # Extract language name from filename (e.g., "english.csv" -> "english")
        language = os.path.splitext(os.path.basename(filepath))[0]
        
        with open(filepath, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            for row in reader:
                # Assume each row contains exactly one sentence in the first column
                if row and row[0].strip():
                    writer.writerow([row[0].strip(), language])
        print(f"Processed: {filepath} -> language: {language}")

print(f"\nAll done! Combined CSV saved to: {output_file}")