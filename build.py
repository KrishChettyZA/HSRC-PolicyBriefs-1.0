# build.py

# This line imports the specific function we need from your main app.py file.
from app import process_all_pdfs
import gc

# This is standard Python practice to make sure the code only runs
# when the script is executed directly.
if __name__ == '__main__':
    print("--- Starting Build Script ---")
    print("Running one-time PDF processing to build the database...")

    # Call the function to process all documents and populate ChromaDB.
    process_all_pdfs()
    
    # Clean up memory after the build process is complete.
    gc.collect()

    print("Database build complete.")
    print("--- Build Script Finished ---")