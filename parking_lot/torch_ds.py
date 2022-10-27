import glob

from torch.utils.data import Dataset
import PIL


class CarParkDS(Dataset):
    def __init__(self, occupied_dir, empty_dir,
                 transform=None, target_transform=None):

        self.imgs = []
        self.transform = transform
        self.target_transform = target_transform

        for img_name in glob.glob(f'{occupied_dir}/*'):
            self.imgs.append((img_name, 1, ))

        for img_name in glob.glob(f'{empty_dir}/*'):
            self.imgs.append((img_name, 0, ))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        # image = read_image(path)
        image = PIL.Image.open(path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
