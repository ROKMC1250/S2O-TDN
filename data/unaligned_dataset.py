import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
from osgeo import gdal


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print(A_path)
        # print(B_path)
        A_img = self.tiff_open(A_path)
        B_img = self.tiff_open(B_path)
        A_img = Image.fromarray(A_img.astype('uint8'))
        B_img = Image.fromarray(B_img.astype('uint8'))
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
    def tiff_open(self, imgPath):

        self.imgPath = imgPath
        self.image = gdal.Open(self.imgPath)
        num_bands = self.image.RasterCount
        # print('original_bands:',num_bands)
        if num_bands == 1:
            band1 = np.expand_dims(np.array(self.image.GetRasterBand(1).ReadAsArray()), axis=2)
            band2 = np.expand_dims(np.array(self.image.GetRasterBand(1).ReadAsArray()), axis=2)
            band3 = np.expand_dims(np.array(self.image.GetRasterBand(1).ReadAsArray()), axis=2)
            self.img_array = np.concatenate([band1, band2, band3], axis=2)
            # self.img_array = band1
        elif num_bands == 3:
            band1 = np.expand_dims(np.array(self.image.GetRasterBand(1).ReadAsArray()) / self.opt.max_value, axis=2)
            band2 = np.expand_dims(np.array(self.image.GetRasterBand(2).ReadAsArray()) / self.opt.max_value, axis=2)
            band3 = np.expand_dims(np.array(self.image.GetRasterBand(3).ReadAsArray()) / self.opt.max_value, axis=2)
            self.img_array = np.concatenate([band1, band2, band3], axis=2)
        else:
            raise ValueError('This function only supprots images with 1 or 3 bands ')

        self.original_path = imgPath
        self.height, self.width, self.bands = self.img_array.shape
        return self.img_array.astype(np.float32)