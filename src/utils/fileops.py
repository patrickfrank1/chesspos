import os


def file_paths_from_directory(directory_path: str, file_ending: str = "") -> list[str]:
    file_list = os.listdir(directory_path)
    files = [f"{directory_path}/{filename}" for filename in file_list if filename.endswith(file_ending)]
    return files
