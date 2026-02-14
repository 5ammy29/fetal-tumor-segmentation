import modal
import os

app = modal.App("upload-dataset")     # Create a Modal app instance

volume = modal.Volume.from_name("dataset", create_if_missing=True)     # Create a modal volume

local_dataset = (     # Path to the local dataset directory
    "/Users/sasha/Documents/fetal-tumor-segmentation/data/dataset"
)

@app.local_entrypoint()
def main():

    with volume.batch_upload() as batch:
        for root, dirs, files in os.walk(local_dataset):
            dirs[:] = [d for d in dirs if not d.startswith(".")]     # Ignore hidden directories
            for f in files:
                if f.startswith("."):
                    continue
                local_path = os.path.join(root, f)
                rel_path = os.path.relpath(local_path, local_dataset)
                batch.put_file(local_path, rel_path)
                
    print("Dataset uploaded successfully")
