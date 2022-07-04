from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader 
import torchvision.transforms as T

class DatasetProcessing(object):
    def __init__(self, image_path, height, width):
        self.path = image_path
        self.h = height 
        self.w = width 

        # read the data from the specified root file and apply the selected transformations
        self.dataset = ImageFolder(root=self.path, transform=T.Compose([
            T.Resize((self.h, self.w)), 
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    def create_data_loader(self, batch_size, shuffle=True):
        dataloader = DataLoader(self.dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=2)
        return dataloader 

    # returns the length of the dataset 
    def __len__(self):
        return len(self.dataset)
    

    
    




