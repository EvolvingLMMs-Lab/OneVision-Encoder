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
    生成一张随机分辨率（min_size~max_size 的正方形）随机像素的 RGB 图像，
    返回 bytes（JPEG 编码）。
    """
    size = random.randint(min_size, max_size)
    # 随机 uint8 图像，形状 (H, W, 3)
    img_array = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')

    # 编码成 JPEG 字节
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return buf.getvalue(), size

def generate_random_label_array(length=10, max_value=10000):
    """
    生成长度为 length 的 label 数组，元素是 [0, max_value) 的随机整数。
    会返回 Python list，但写入 record 时会转成 float32 的 numpy array。
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

    # 如果存在旧文件，先删掉
    for p in [rec_path, idx_path]:
        if os.path.exists(p):
            os.remove(p)

    # 创建 MXIndexedRecordIO
    record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'w')

    for i in range(num_samples):
        # 生成随机图片和对应的 label
        img_bytes, size = generate_random_image(min_res, max_res)
        label_list = generate_random_label_array(label_length, max_label_value)

        # MXRecordIO 的 label 会被转成 float32 数组
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
