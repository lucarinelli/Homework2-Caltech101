from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    
    dataset = []
    
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        
        if self.split == 'train':
            filepath = root+'/'+'train.txt'
        elif self.split == 'test':
            filepath = root+'/'+'test.txt'

        if not os.path.isfile(filepath):
            print("File path {} does not exist. Error!!!".format(filepath))
            return
  
        
        prev_target_name = None
        target_number = 0
        self.dataset = []
        
        datadir = '101_ObjectCategories'
        
        with open(filepath) as fp:
            for line in fp:
                target_name = line.split('/')[0]
                
                if target_name == 'BACKGROUND_Google':
                    continue
                
                if prev_target_name is None:
                    prev_target_name = target_name
                elif prev_target_name != target_name:
                    prev_target_name = target_name
                    target_number = target_number + 1
                
                self.dataset.append(pil_loader(root+'/'+datadir+'/'+line), target_number)

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.dataset[index] # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.dataset) # Provide a way to get the length (number of elements) of the dataset
        return length
