# -- rand nums --
import random

# -- linalg --
import torch as th
import numpy as np

# -- image pair dataset --
import random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class ImagePairDataSet(data.Dataset):

    def __init__(self, block_w, images_a=None, images_b=None, transform=None, stride=1):
        self.transform = transform
        self.images_a = images_a
        self.images_b = images_b
        self.im_n, _, self.im_h, self.im_w = self.images_a.shape
        self.stride = stride
        self.block_w = block_w
        self.blocks_in_image_h = (self.im_h - self.block_w) // stride + 1
        self.blocks_in_image_w = (self.im_w - self.block_w) // stride + 1
        # print("ImagePairDataSet Info")
        # print(self.images_a.shape)
        # print(self.im_h,self.block_w)
        # print(self.blocks_in_image_h)
        self.len = self.im_n * self.blocks_in_image_h * self.blocks_in_image_w
        self.ishift = ShiftImageValues()

        if len(self.images_a.shape) < 4:
            self.images_a = np.expand_dims(self.images_a, axis=3)
            self.images_b = np.expand_dims(self.images_b, axis=3)

    def augment(self,img_a,img_b):

        # Random horizontal flipping
        if random.random() > 0.5:
            img_a = TF.hflip(img_a)
            img_b = TF.hflip(img_b)

        # Random vertical flipping
        if random.random() > 0.5:
            img_a = TF.vflip(img_a)
            img_b = TF.vflip(img_b)

        # Random transpose
        if random.random() > 0.5:
            img_a = img_a.transpose(-1,-2)
            img_b = img_b.transpose(-1,-2)

        # Transform to tensor
        # img_a = self.ishift(img_a)
        # img_b = self.ishift(img_b)

        return img_a,img_b


    def __getitem__(self, item):
        im, row, col = np.unravel_index(item, (self.im_n, self.blocks_in_image_h,
                                               self.blocks_in_image_w))
        row *= self.stride
        col *= self.stride
        sample_a = self.images_a[im, :,row:row + self.block_w, col:col + self.block_w]
        sample_b = self.images_b[im, :,row:row + self.block_w, col:col + self.block_w]
        sample_a,sample_b = self.augment(sample_a,sample_b)

        return sample_a,sample_b

    def __len__(self):
        return self.len

class RandomTranspose(object):
    """Applies transpose the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be transposed.

        Returns:
            PIL Image: Randomly transposed image.
        """
        if random.random() < self.p:
            if not isinstance(img, Image.Image):
                raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

            return img.transpose(Image.TRANSPOSE)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ShiftImageValues(object):
    def __call__(self, img):
        return (img - 0.5) / 0.5

    def __repr__(self):
        return self.__class__.__name__
