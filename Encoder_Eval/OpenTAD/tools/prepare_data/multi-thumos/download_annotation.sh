DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for Multi-THUMOS
gdown --folder https://drive.google.com/drive/folders/1LtSwg5UhmWvZ5VEwqXQmNf36t-pWXXdU -O $DATA_DIR --folder