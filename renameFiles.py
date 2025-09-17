import os
from pathlib import Path

def rename_files_in_documents_folder():
    """
    This script renames all files in the 'documents' subfolder of the
    current directory by replacing spaces with hyphens.
    """
    # Get the directory where the script is located and append '/documents'
    # This makes the script runnable from the root of your project.
    current_dir = Path(__file__).parent
    documents_dir = current_dir / "documents"

    # Check if the documents directory actually exists
    if not documents_dir.is_dir():
        print(f"Error: The directory '{documents_dir}' was not found.")
        print("Please make sure you are running this script from your main project folder (e.g., ARC1.1).")
        return

    print(f"Scanning for files in: {documents_dir}\n")
    renamed_count = 0

    # Loop through every file in the documents directory
    for filename in os.listdir(documents_dir):
        # Check if the filename contains a space
        if ' ' in filename:
            # Create the new filename by replacing spaces with hyphens
            new_filename = filename.replace(' ', '-')
            
            # Get the full old and new file paths
            old_filepath = documents_dir / filename
            new_filepath = documents_dir / new_filename
            
            # Rename the file
            os.rename(old_filepath, new_filepath)
            
            # Print a confirmation message
            print(f"Renamed: '{filename}'  ->  '{new_filename}'")
            renamed_count += 1

    if renamed_count == 0:
        print("No files with spaces were found to rename.")
    else:
        print(f"\n✅ Success! Renamed {renamed_count} file(s).")

# This ensures the function runs when the script is executed
if __name__ == "__main__":
    rename_files_in_documents_folder()
