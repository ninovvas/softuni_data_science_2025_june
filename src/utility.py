import os

def save_file(data, name, directory="../data"):
    """Saves a pandas dataframe to a csv file in the specified directory

    Parameters:
        - data : pandas.DataFrame. The DataFrame to be saved.
        - name : str. The filename for the CSV (e.g., 'output.csv').
        - directory : str, optional. The directory where the file should be saved. Defaults to "../data".
        - Returns: None
    """
    
    file_path = os.path.join(directory, name)
    
    # create the directory if it doesn't exist
    os.makedirs(directory, exist_ok = True)
    data.to_csv(file_path, index = False)
