import sys, pathlib
from pathlib import Path

def project_path():
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    return project_dir

def add_project_path():
    """
    add proj dir ro path for import file resolution
    """
    project_dir = project_path()
    sys.path.append(str(project_dir))

# train/test data path
def data_path():
    project_dir = project_path()
    return project_dir / "data"

# dir to which best trained classifier models are saved to, alongside it's dictionaries
def model_output_path():
    project_dir = project_path()
    return project_dir / "output" / "models"

def huggingface_model_path():
    project_dir = project_path()
    return project_dir / "huggingface_model"

def fine_tuned_model_path():
    project_dir = project_path()
    return project_dir / "fine_tuned_models"

def classifier_model_load_path(load_attribute = "591_68_75"):
    return model_output_path() / f"classifier_{load_attribute}.pt"