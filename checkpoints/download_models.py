import requests

# URL файла для скачивания
url = 'https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth'
# Имя файла, под которым он будет сохранен
file_name = 'XMem.pth'
path = 'checkpoints'
# Базовые URL-адреса
BASE_URL_PT_1 = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
BASE_URL_YAML_1 = (
    "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1"
)


# Словарь для чекпоинтов (.pt)
CHECKPOINTS_PT_2_1 = {
    f"{path}/sam2.1_hiera_tiny.pt": f"{BASE_URL_PT_1}/sam2.1_hiera_tiny.pt",
    f"{path}/sam2.1_hiera_small.pt": f"{BASE_URL_PT_1}/sam2.1_hiera_small.pt",
    f"{path}/sam2.1_hiera_base_plus.pt": f"{BASE_URL_PT_1}/sam2.1_hiera_base_plus.pt",
    f"{path}/sam2.1_hiera_large.pt": f"{BASE_URL_PT_1}/sam2.1_hiera_large.pt",
}

# Словарь для конфигурационных файлов (.yaml)
# CHECKPOINTS_YAML_1 = {
#     f"{path}/sam2.1_hiera_t.yaml": f"{BASE_URL_YAML_1}/sam2.1_hiera_t.yaml",
#     f"{path}/sam2.1_hiera_s.yaml": f"{BASE_URL_YAML_1}/sam2.1_hiera_s.yaml",
#     f"{path}/sam2.1_hiera_b+.yaml": f"{BASE_URL_YAML_1}/sam2.1_hiera_b+.yaml",
#     f"{path}/sam2.1_hiera_l.yaml": f"{BASE_URL_YAML_1}/sam2.1_hiera_l.yaml",
# }

BASE_URL_PT = "https://dl.fbaipublicfiles.com/segment_anything_2/072824"
BASE_URL_YAML = "https://raw.githubusercontent.com/Segment-Anything/segment-anything-2/main/sam2_configs"
# https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2

CHECKPOINTS_PT_2 = {
    f"{path}/sam2_hiera_tiny.pt": f"{BASE_URL_PT}/sam2_hiera_tiny.pt",
    f"{path}/sam2_hiera_small.pt": f"{BASE_URL_PT}/sam2_hiera_small.pt",
    f"{path}/sam2_hiera_base_plus.pt": f"{BASE_URL_PT}/sam2_hiera_base_plus.pt",
    f"{path}/sam2_hiera_large.pt": f"{BASE_URL_PT}/sam2_hiera_large.pt",
}

# Словарь для конфигурационных файлов (.yaml)
# CHECKPOINTS_YAML = {
#     f"{path}/sam2_hiera_t.yaml": f"{BASE_URL_YAML}/sam2_hiera_t.yaml",
#     f"{path}/sam2_hiera_s.yaml": f"{BASE_URL_YAML}/sam2_hiera_s.yaml",
#     f"{path}/sam2_hiera_b+.yaml": f"{BASE_URL_YAML}/sam2_hiera_b+.yaml",
#     f"{path}/sam2_hiera_l.yaml": f"{BASE_URL_YAML}/sam2_hiera_l.yaml",
# }


def download_checkpoint(url, filename):
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Проверка на ошибки HTTP

        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"{filename} downloaded successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {filename} from {url}. Error: {e}")
        exit(1)


if __name__ == "__main__":
    download_checkpoint(url, f'{path}/{file_name}')
    # Сначала скачиваем .pt файлы
    for filename, url in CHECKPOINTS_PT_2.items():
        download_checkpoint(url, filename)

    for filename, url in CHECKPOINTS_PT_2_1.items():
        download_checkpoint(url, filename)

    print("All checkpoints and configuration files are downloaded successfully.")
