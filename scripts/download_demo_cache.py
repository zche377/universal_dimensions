from dotenv import load_dotenv
load_dotenv()

import os
from osfclient.api import OSF
from osfclient.models.file import Folder
from bonner.caching import BONNER_CACHING_HOME

DEMO_DATA_ID = "wvj3s"

PATH_DICT = {
    "bonner-caching": BONNER_CACHING_HOME,
    "bonner-models": os.getenv("BONNER_MODELS_HOME"),
    # "datasets": os.getenv("BONNER_DATASETS_HOME"),
}


def download_osf_project(project_id, path_dict):
    osf = OSF()
    project = osf.project(project_id)
    storage = project.storage('osfstorage')
    
    for top_level_item in storage.folders:
        if isinstance(top_level_item, Folder) and top_level_item.name in path_dict:
            local_destination = path_dict[top_level_item.name]
            if not os.path.exists(local_destination):
                os.makedirs(local_destination)
            
            print(f"Downloading contents of folder '{top_level_item.name}' to '{local_destination}'")
            download_folder(top_level_item, local_destination)
            
def download_folder(folder, local_path):
    for item in folder.folders:
        item_path = os.path.join(local_path, item.name)
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        download_folder(item, item_path)
    for item in folder.files:
        item_path = os.path.join(local_path, item.name)
        print(f"Downloading file '{item.name}' to '{item_path}'")
        with open(item_path, 'wb') as f:
            item.write_to(f)


if __name__ == "__main__":
    download_osf_project(DEMO_DATA_ID, PATH_DICT)