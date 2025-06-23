import PowerFactory_Control.Get_Nested_Folder as get_nested_folder
import sys
#import tkinter as tk
#from tkinter import filedialog

# keep for later
'''
def select_directory():
    # Initialize Tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()

    # Open a dialog to select a directory
    directory = filedialog.askdirectory(title="Select Python Directory")
    
    # Clean up the Tkinter instance
    root.destroy()
    return directory

# Ask the user to select a directory
selected_dir = select_directory()

if selected_dir:
    # Append the selected directory to sys.path
    sys.path.append(selected_dir)
    print(f"Appended '{selected_dir}' to sys.path")
else:
    print("No directory was selected.")
'''


# Power Factory Setup
def powerfactory_setup(pf_data):
    print('Setting up PowerFactory')

    selected_dir = r'C:\Program Files\DIgSILENT\PowerFactory 2024 SP4\Python\3.9'  

    # Append the selected directory to sys.path
    sys.path.append(selected_dir)
    print(f"Appended '{selected_dir}' to sys.path")

    # Import PowerFactory module
    import powerfactory 
    ############### Power Factory Connection With Application Startup Failure Exception ###############
    
    try:
        #username = None
        #password = None
        #command_line_args = r'/ini "C:\Users\JamesThornton\OneDrive - Blake Clough consulting\Desktop\Power Factory Configurations\PowerFactory_3.ini"'
        #pf_data.app = powerfactory.GetApplicationExt(username, password, command_line_args)
        pf_data.app = powerfactory.GetApplicationExt()
    except powerfactory.ExitError as error:
        print(error)
        print('error.code = %d' % error.code)
        sys.exit(error.code)  # or handle the error as needed

    ############### Activating Power Factory Project ###############

    pf_data.user = pf_data.app.GetCurrentUser()
    pf_data.projects = pf_data.user.GetContents(pf_data.project_name + '.IntPrj')[0]
    pf_data.projects.Activate()
    pf_data.project = pf_data.projects
    active_study_case = pf_data.app.GetActiveStudyCase()
    if active_study_case is not None:
        active_study_case.Activate()
    print(pf_data.projects)

