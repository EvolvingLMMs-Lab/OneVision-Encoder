import json
import os

from PIL import Image
from src.open_clip import tokenize
from torch.utils.data import DataLoader, Dataset

data_root = f"{os.environ['LINEAR_PROBE_ROOT']}/flicker30K/flickr30k-images"
json_file = f"{os.environ['LINEAR_PROBE_ROOT']}/flicker30K/flickr30k_test.json"
target_transform = None

class flickr_text_dataset(Dataset):
    def __init__(self, text_data, tokenizer):
        self.tokenizer = tokenizer
        self.caption = text_data

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, index):
        text_data = self.caption[index]
        text_data = 'a photo of ' + text_data
        text_token = self.tokenizer(text_data)
        return text_token


class flickr_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root):
        self.ann_file = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.caption = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        for num, line in enumerate(self.ann_file):
            image_name = line['image'].split('/')[1]
            image_path = os.path.join(data_root, image_name)
            self.image.append(image_path)
            self.caption += line['caption']
            for i in range(5*num, 5*(num+1)):
                self.txt2img[i] = num
                if num not in self.img2txt.keys():
                    self.img2txt[num] = [i]
                else:
                    self.img2txt[num].append(i)
                 
    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index


def get_loader_image(batch_size, preprocess):
    valid_dataset = flickr_dataset(json_file, preprocess, data_root)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False)
    return valid_dataloader, valid_dataset.txt2img, valid_dataset.img2txt


def get_loader_text(batch_size, preprocess):
    valid_dataset = flickr_dataset(json_file, preprocess, data_root)
    text_data = valid_dataset.caption
    text_dataset = flickr_text_dataset(text_data, tokenize)
    valid_dataloader = DataLoader(text_dataset, batch_size = batch_size, shuffle=False)
    return valid_dataloader