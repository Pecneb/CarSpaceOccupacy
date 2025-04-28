import yaml

def read_parameters(file_path):
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary containing the parameters from the YAML file.
    """
    try:
        with open(file_path, 'r') as file:
            parameters = yaml.safe_load(file)
        return parameters
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")

# Example usage
if __name__ == "__main__":
    file_path = "/Users/bencegaborpeter/gitclones/CarSpaceOccupacy/config/red_kia_parameters.yaml"
    parameters = read_parameters(file_path)
    if parameters:
        print(parameters)