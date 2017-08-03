from torchvision import datasets, transforms, utils
import torch
import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import pickle
import matplotlib.pyplot as plt


class LoadLIPDataset(Dataset):
    def __init__(self,list_file, seg_dir, root_dir, transform=None):
        """
        Args:
            seg_dir (string): Directory with all the segemented images
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(list_file, 'rb') as fp:
            self.list_file = pickle.load(fp)
        self.seg_dir = seg_dir
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.list_file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.list_file[idx])
        image = io.imread(img_name)
        seg_name = os.path.join(self.seg_dir, self.list_file[idx])
        segment = io.imread(seg_name.replace('jpg','png'));
        sample = {'image': image, 'segment': segment}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, segment = sample['image'], sample['segment']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        segment = transform.resize(segment, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively


        return {'image': img, 'segment': segment}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, segment = sample['image'], sample['segment']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        #Segments are black and white png images

        return {'image': torch.from_numpy(image),
                'segment': torch.from_numpy(segment)}


kwargs = {
			'num_workers': 1,
			'batch_size': 4,
			'shuffle': True,
			'pin_memory': False}

data_transforms = {
			'train': transforms.Compose([
                Rescale((321,321)),
               ToTensor()

			]),
			'val': transforms.Compose([
                Rescale((321,321)),
                ToTensor()

			])
		}


#For testing
data_dir ='../data/LIP/TrainVal_images/TrainVal_images'
s_dir ='../data/LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations'
list_dir = '../data/LIP/TrainVal_images/TrainVal_images'

dsets = {x: LoadLIPDataset(list_file=os.path.join(list_dir, x + '_list'),
                                    root_dir=os.path.join(data_dir, x + '_images'), seg_dir=os.path.join(s_dir, x + '_segmentations'),
                                     transform=data_transforms[x]) for x in ['train', 'val']}

train_loader = torch.utils.data.DataLoader(dsets["train"], **kwargs)
val_loader = torch.utils.data.DataLoader(dsets["val"], **kwargs)

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, segment_batch = \
            sample_batched['image'], sample_batched['segment']

    #print(segment_batch[0].size())
    grid = utils.make_grid(images_batch)
    grid2 = utils.make_grid(segment_batch)
    print(grid2.size())
    #plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.imshow(grid2)

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['segment'].size())


    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
