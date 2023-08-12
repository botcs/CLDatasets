import argparse
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


class CLDatasets:
    """
    A class for downloading datasets from Google Cloud Storage.
    """

    def __init__(self, dataset: str, src_directory: str, out_directory_root: str):
        """
        Initialize the CLDatasets object.

        Args:
            dataset (str): The name of the dataset to download.
            directory (str): The directory where the dataset will be saved.
        """
        if dataset not in ['CGLM', 'CLOC', 'ImageNet2K']:
            print("Dataset not found!")
            return
        else:
            self.unzip_data_files(src_directory+f"/{dataset}/data", out_directory_root)

    def unzip_data_files(self, src_directory: str, out_directory_root: str) -> None:
        """
        Extracts the contents of zip files in a directory into nested folders.

        Args:
            directory: The path to the directory containing the zip files.

        Returns:
            None
        """

        zip_files = [file for file in os.listdir(
            src_directory) if file.endswith('.zip')]

        os.makedirs(out_directory_root, exist_ok=True)

        def extract_single_zip(zip_file: str) -> None:

            zip_path = os.path.join(src_directory, zip_file)
            output_dir = os.path.join(
                out_directory_root, os.path.splitext(os.path.basename(zip_file))[0])

            os.makedirs(output_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

        with ThreadPoolExecutor() as executor, tqdm(total=len(zip_files)) as pbar:
            futures_list = []
            for zip_file in zip_files:
                future = executor.submit(extract_single_zip, zip_file)
                future.add_done_callback(lambda p: pbar.update(1))
                futures_list.append(future)

            # Wait for all tasks to complete
            for future in futures_list:
                future.result()

        # Remove zip files
        # remove_command = f"rm {self.directory}/{self.dataset}/data/*.zip"
        # os.system(remove_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download datasets from Google Cloud Storage.')
    parser.add_argument('--dataset', type=str, default='CGLM',
                        help='The name of the dataset to download.')
    parser.add_argument('--src-directory', type=str,
                        help='The directory where the zip files are located.')
    parser.add_argument('--out-directory', type=str,
                        help='The directory where the dataset will be saved.')

    args = parser.parse_args()

    gcp_cl_datasets = CLDatasets(
        dataset=args.dataset,
        src_directory=args.src_directory,
        out_directory_root=args.out_directory,
    )
