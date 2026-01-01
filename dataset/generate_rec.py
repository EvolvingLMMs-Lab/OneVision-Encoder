import numpy as np
np.bool = np.bool_


import os
import random
import mxnet as mx
import numpy as np
from PIL import Image
import io

def generate_random_image(min_size=256, max_size=1024):
    """
    Generate an RGB image with random resolution (square of min_size~max_size) and random pixels,
    returns bytes (JPEG encoded).
    """
    size = random.randint(min_size, max_size)
    # Random uint8 image, shape (H, W, 3)
    img_array = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')

    # Encode to JPEG bytes
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return buf.getvalue(), size

def generate_random_label_array(length=10, max_value=10000):
    """
    Generate a label array of length with random integers in [0, max_value).
    Returns Python list, but converts to float32 numpy array when writing to record.
    """
    return [random.randint(0, max_value - 1) for _ in range(length)]

def main():
    output_prefix = "fake_data"
    rec_path = output_prefix + ".rec"
    idx_path = output_prefix + ".idx"

    num_samples = 10000
    min_res = 256
    max_res = 1024
    label_length = 10
    max_label_value = 10000

    # If old files exist, delete them first
    for p in [rec_path, idx_path]:
        if os.path.exists(p):
            os.remove(p)

    # Create MXIndexedRecordIO
    record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'w')

    for i in range(num_samples):
        # Generate random image and corresponding label
        img_bytes, size = generate_random_image(min_res, max_res)
        label_list = generate_random_label_array(label_length, max_label_value)

        # MXRecordIO's label will be converted to float32 array
        header = mx.recordio.IRHeader(
            flag=0,
            label=np.array(label_list, dtype=np.float32),
            id=i,
            id2=0
        )
        packed = mx.recordio.pack(header, img_bytes)
        record.write_idx(i, packed)

        if (i + 1) % 1000 == 0:
            print(f"Written {i + 1}/{num_samples} samples, last image size: {size}x{size}")

    print(f"Done! Wrote {num_samples} samples to {rec_path} (index: {idx_path}).")

if __name__ == "__main__":
    main()
