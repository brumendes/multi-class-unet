from pathlib import Path
import pandas as pd
import shutil

# Read the dataset
df = pd.read_csv(Path(r"C:\Users\admin\Documents\quibpca\datasets\prostate_dataset_test.csv"))

# Convert Linux paths to Windows paths (replace forward slashes and mount point)
df['path'] = df['Image'].str.replace('/media/mendes/BACKUP1', 'G:', regex=False)
df['path'] = df['path'].str.replace('/', '\\', regex=False)

# Create destination paths
df['new_path'] = df['path'].str.replace('G:', r'C:\Users\admin\Documents', regex=False)

# Copy directories and their contents
for i, row in df.iterrows():
    src = Path(row['path'])
    dst = Path(row['new_path'])

    print(f"Copying {src} -> {dst}")
    
    if not src.exists():
        print(f"Source not found: {src}")
        continue
        
    try:
        # Create destination directory if it doesn't exist
        dst.mkdir(parents=True, exist_ok=True)
        
        # Copy directory contents
        for file in src.rglob('*'):  # Use rglob to include subdirectories
            if file.is_file():  # Only copy files
                rel_path = file.relative_to(src)
                dst_file = dst / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, dst_file)  # Use copy2 to preserve metadata
                
        print(f"Successfully copied: {src} -> {dst}")
        
    except Exception as e:
        print(f"Error processing {src}: {str(e)}")

print("\nProcessing completed!")
