DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for HACS-1.1
gdown --folder https://drive.google.com/drive/folders/1G0fv3CXDQLsoJgaSyBbIi0_KGu6cKjH2 -O $DATA_DIR --folder