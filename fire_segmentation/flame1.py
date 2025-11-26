from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import random

# Patch wise

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as T


class FireDataset512(Dataset):
    def __init__(self, images_dir, masks_dir, image_list=None, transform=None, resize=None):
        """
        Initializes the dataset by reading the provided list of image file names.

        Args:
        - images_dir (str): Directory with all the images.
        - masks_dir (str): Directory with all the masks.
        - image_list (list, optional): List of image file names to use (without directory path).
        - transform (callable, optional): A function/transform that takes in an image and mask and returns transformed versions.
        - resize (tuple, optional): Tuple indicating the desired size (height, width) for resizing images and masks.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.resize = resize  # Store the resize parameter
        self.images = image_list if image_list is not None else os.listdir(images_dir)

        NUM_CLASS = 1
        IN_CHANNELS = 3
        # Define resizing transformation if needed
        if self.resize is not None:
            self.resize_transform = T.Resize(self.resize, interpolation=T.InterpolationMode.BILINEAR)
            self.mask_resize_transform = T.Resize(self.resize, interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
      Loads and returns a sample from the dataset at the given index.

      Args:
      - index (int): Index of the image to retrieve.

      Returns:
      - image (Tensor): The transformed image.
      - mask (Tensor): The corresponding transformed mask.
      """
        # Get the image and mask file names
        img_name = self.images[index]

        # Paths for the image and corresponding mask
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace('.jpg', '.png'))

        # Load image and mask
        image = Image.open(img_path).convert("RGB")  # Convert image to RGB
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Resize the image and mask if resize is specified
        if self.resize is not None:
            image = self.resize_transform(image)  # Apply resizing to the image
            mask = self.mask_resize_transform(mask)  # Apply resizing to the mask


        # Apply additional transformations (if provided)
        if self.transform is not None:
            # Albumentations typically returns a dictionary with keys 'image' and 'mask'
            image, mask = self.apply_transforms(image, mask)

        # Convert to NumPy arrays
        image = np.array(image) / 255.0
        mask = np.array(mask, dtype=np.float32)

        # Convert the mask: Binary conversion (255 to 1.0)
        #mask[mask < 127.0] = 0.0
        mask[mask >= 127.0] = 1.0

        # Convert image and mask to PyTorch tensors
        image = torch.from_numpy(image).float().permute(2, 0, 1)  # From HWC to CHW
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dimension to mask

        return image, mask

    def apply_transforms(self, image, mask):
        """
        Apply transforms to both the image and the mask in a consistent way.
        Some transforms are only applied to the image (like ColorJitter),
        while others (like geometric transformations) must be applied to both.
        """
        for t in self.transform.transforms:
            if isinstance(t, T.RandomHorizontalFlip):
                # Apply RandomHorizontalFlip consistently
                if random.random() < 0.5:
                    image = F.hflip(image)
                    mask = F.hflip(mask)

            elif isinstance(t, T.RandomRotation):
                # Apply RandomRotation consistently
                angle = t.get_params(t.degrees)  # Get the same random angle for both
                image = F.rotate(image, angle)
                mask = F.rotate(mask, angle)

            # Add any other transformations you want to handle similarly

        return image, mask



class FireDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_list=None, transform=None, resize=(320, 224), stride=(320, 224)):
        # Half Stride: 160x112
        # Quarter Stride: 80x56
        # Further Stride: 40x28
        """
        Initializes the dataset for patch-based approach with a sliding window.

        Args:
        - images_dir (str): Directory with all the images.
        - masks_dir (str): Directory with all the masks.
        - image_list (list, optional): List of image file names to use (without directory path).
        - transform (callable, optional): A function/transform that takes in an image and mask and returns transformed versions.
        - patch_size (tuple, optional): Tuple (height, width) for the size of the patches.
        - stride (tuple, optional): Tuple (height_stride, width_stride) for sliding window step size.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.patch_size = resize  # Store the patch size
        self.stride = stride  # Store the stride size
        self.images = image_list if image_list is not None else os.listdir(images_dir)

        # Assuming all images are of the same size, we use img_size to compute patches
        img_width, img_height = (3840, 2160)
        self.patches_w = (img_width - self.patch_size[0]) // self.stride[0] + 1
        self.patches_h = (img_height - self.patch_size[1]) // self.stride[1] + 1

        # Compute the total number of patches
        self.patches_per_image = self.patches_w * self.patches_h
        self.total_patches = self.patches_per_image * len(self.images)

    def __len__(self):
        """
        Returns the total number of patches in the dataset.
        """
        return self.total_patches

    def get_patch(self, image, mask, top, left):
        """
        Extract a patch from the given image and mask starting at (top, left).

        Args:
        - image (PIL Image): The input RGB image.
        - mask (PIL Image): The corresponding grayscale mask.
        - top (int): The top-left corner's y-coordinate.
        - left (int): The top-left corner's x-coordinate.

        Returns:
        - patch_image (PIL Image): The cropped RGB image patch.
        - patch_mask (PIL Image): The cropped mask patch.
        """
        patch_width, patch_height = self.patch_size

        # Crop the image and mask
        patch_image = image.crop((left, top, left + patch_width, top + patch_height))
        patch_mask = mask.crop((left, top, left + patch_width, top + patch_height))

        return patch_image, patch_mask

    def __getitem__(self, index):
        """
        Loads and returns a patch from the dataset at the given index.

        Args:
        - index (int): Index of the patch to retrieve.

        Returns:
        - patch_image (Tensor): The transformed patch image.
        - patch_mask (Tensor): The corresponding transformed patch mask.
        """
        # Find which image this index corresponds to
        img_index = index // self.patches_per_image  # Get the image index
        patch_index = index % self.patches_per_image  # Get the patch index within that image

        # Get the row and column of the patch
        row = patch_index // self.patches_w
        col = patch_index % self.patches_w

        # Get the image and mask file names
        img_name = self.images[img_index]

        # Paths for the image and corresponding mask
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace('.jpg', '.png'))

        # Load image and mask
        image = Image.open(img_path).convert("RGB")  # Convert image to RGB
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Calculate the top and left positions of the patch
        top = row * self.stride[1]
        left = col * self.stride[0]

        # Extract the patch
        patch_image, patch_mask = self.get_patch(image, mask, top, left)

        # Apply additional transformations (if provided)
        if self.transform is not None:
            patch_image, patch_mask = self.apply_transforms(image=patch_image, mask=patch_mask)

        # Convert to NumPy arrays
        patch_image = np.array(patch_image) / 255.0
        patch_mask = np.array(patch_mask, dtype=np.float32)

        # Convert the mask: Binary conversion (255 to 1.0)
        #patch_mask[patch_mask < 127.0] = 0.0
        patch_mask[patch_mask >= 127.0] = 1.0

        # Convert image and mask to PyTorch tensors
        patch_image = torch.from_numpy(patch_image).float().permute(2, 0, 1)  # From HWC to CHW
        patch_mask = torch.from_numpy(patch_mask).float().unsqueeze(0)  # Add channel dimension to mask

        return patch_image, patch_mask

    def apply_transforms(self, image, mask):
        """
        Apply transforms to both the image and the mask in a consistent way.
        Some transforms are only applied to the image (like ColorJitter),
        while others (like geometric transformations) must be applied to both.
        """
        for t in self.transform.transforms:
            if isinstance(t, T.RandomHorizontalFlip):
                # Apply RandomHorizontalFlip consistently
                if random.random() < 0.5:
                    image = T.hflip(image)
                    mask = T.hflip(mask)

            elif isinstance(t, T.RandomRotation):
                # Apply RandomRotation consistently
                angle = t.get_params(t.degrees)  # Get the same random angle for both
                image = T.rotate(image, angle)
                mask = T.rotate(mask, angle)

            # Add any other transformations you want to handle similarly

        return image, mask


def load_dataset(data_path='../../FLAME1', patch_wise=False, batch_size=32, n_worker=0):
    images_dir = data_path + '/Images'
    mask_dir = data_path + '/Masks'
    images_list = os.listdir(images_dir)

    # Split the dataset into train, validation, and test
    train_images, test_images = train_test_split(images_list, test_size=0.15, random_state=42)
    train_images, val_images = train_test_split(train_images, test_size=0.15, random_state=42)

    print(f'Training samples: {len(train_images)} Validation samples: {len(val_images)} Test samples: {len(test_images)}')

    # Create a custom dataset class for handling specific subsets of images
    # Orginal: torch.Size([8, 3, 2160, 3840]) torch.Size([8, 1, 2160, 3840])
    # Previous: resize_fmt = (240, 320)
    # Proposed: resize_fmt = (224, 320)
    num_worker = n_worker

    # Deep Lab v3+ Augmentations
    '''transform = T.Compose([
        T.RandomPerspective(distortion_scale=.3),
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.1),
                       scale=(0.5, 1.5))]
    )'''

    # Baseline: Customized UNet Augmentations
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(10)]
    )

    if not patch_wise:
        resize_fmt = (512, 512)
        train_dataset = FireDataset512(images_dir, mask_dir, image_list=train_images, transform=transform,
                                       resize=resize_fmt)
        val_dataset = FireDataset512(images_dir, mask_dir, image_list=val_images, transform=None,
                                     resize=resize_fmt)
        test_dataset = FireDataset512(images_dir, mask_dir, image_list=test_images, transform=None,
                                      resize=resize_fmt)

    else:
        resize_fmt = (224, 320)
        train_dataset = FireDataset(images_dir, mask_dir, image_list=train_images, transform=transform,
                                    resize=resize_fmt)
        val_dataset = FireDataset(images_dir, mask_dir, image_list=val_images, transform=None,
                                  resize=resize_fmt)
        test_dataset = FireDataset(images_dir, mask_dir, image_list=test_images, transform=None,
                                   resize=resize_fmt)

    # Create DataLoader for each dataset

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    return train_dataloader, valid_dataloader, test_dataloader
