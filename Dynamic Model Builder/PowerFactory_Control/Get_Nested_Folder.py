
# Retrieve the grid folder location from powerfactory
def get_nested_folder(pf_data, path):
    # Start with a folder you're sure exists; adjust as needed for your project
    current_folder = pf_data.app.GetActiveProject()
    print(f"Starting folder: {current_folder.loc_name}")  # Debug print

    for folder_name in path:
        print(f"Looking for: {folder_name} in {current_folder.loc_name}")  # Debug print
        found_folders = current_folder.GetContents(folder_name)
        
        if not found_folders:
            raise Exception(f"Folder '{folder_name}' not found in '{current_folder.loc_name}'")
            logging.error(f"Folder '{folder_name}' not found in '{current_folder.loc_name}'")
        
        current_folder = found_folders[0]  # Assuming the first match is the correct one
        print(f"Found folder: {current_folder.loc_name}")  # Debug print

    return current_folder
