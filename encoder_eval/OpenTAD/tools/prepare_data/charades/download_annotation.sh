DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for Charades
gdown --folder https://drive.google.com/drive/folders/1AOG2n_4X5_TipZXwEOLEJI2zJr7_7xiq -O $DATA_DIR --folder