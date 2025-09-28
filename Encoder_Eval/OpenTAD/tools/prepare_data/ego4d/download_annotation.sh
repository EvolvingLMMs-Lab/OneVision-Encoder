DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for Ego4D-MQ
gdown https://drive.google.com/drive/folders/1Yha2NmDL-llmcJ3t-vUBUj2EkfWmoBaN -O $DATA_DIR --folder
