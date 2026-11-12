import torch
from PIL import Image
from torch.utils.data import Dataset


class AgeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe.loc[index, "image_path"]
        age = self.dataframe.loc[index, "age"]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        age = torch.tensor(age, dtype=torch.float32)

        return image, age
