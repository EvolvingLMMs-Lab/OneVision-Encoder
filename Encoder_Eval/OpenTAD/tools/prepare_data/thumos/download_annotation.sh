DATA_DIR="../../../data/"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for THUMOS-14
gdown --folder https://drive.google.com/drive/folders/1ee-ZeGXK5U78R5tc528-hu9C-goUmt8r -O $DATA_DIR --folder