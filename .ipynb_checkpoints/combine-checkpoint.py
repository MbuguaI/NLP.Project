import csv
import os

data_folder = "data"
output_file = os.path.join(data_folder, "combined.csv")

# Only these files will be combined
language_files = {
    "english.csv": "english",
    "sheng.csv": "sheng",
    "swahili.csv": "swahili",
    "luo.csv": "luo"
}

os.makedirs(data_folder, exist_ok=True)

with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
    writer.writerow(["sentence", "language"])

    for filename, language in language_files.items():
        filepath = os.path.join(data_folder, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue

        with open(filepath, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            for row in reader:
                if row and row[0].strip():
                    writer.writerow([row[0].strip(), language])
        print(f"Processed: {filename} -> language: {language}")

print(f"\nAll done! Combined CSV saved to: {output_file}")