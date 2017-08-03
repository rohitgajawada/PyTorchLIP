from torchvision import datasets, transforms, utils
import torch
import os
import utils
import torch
import gen_pytorch as gp
from torch.utils.data import Dataset, DataLoader
from data_class import LoadLIPDataset, Rescale, ToTensor

class LoadMNIST():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': 4,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			datasets.MNIST('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,))
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			datasets.MNIST('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
			  **kwargs)

class LoadCIFAR10():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': 4,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
std=[x/255.0 for x in [63.0, 62.1, 66.7]])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
std=[x/255.0 for x in [63.0, 62.1, 66.7]])
					   ])),
		  **kwargs)

class LoadCIFAR100():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': 4,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR100('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.RandomCrop(32, padding=4),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
std=[x/255.0 for x in [68.2, 65.4, 70.4]])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			datasets.CIFAR100('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize(mean=[x/255.0 for x in [129.3, 124.1, 112.4]],
std=[x/255.0 for x in [68.2, 65.4, 70.4]])
					   ])),
		  **kwargs)

class LoadTuberlin():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': 4,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		data_transforms = {
			'train': transforms.Compose([
				transforms.RandomSizedCrop(225),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				utils.ToGray(),
				#Normalization pending
				transforms.Normalize([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
			]),
			'val': transforms.Compose([
				transforms.Scale(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				utils.ToGray(),
				#Normalization pending
				transforms.Normalize([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
			])
		}

		data_dir = '../data/tuberlin'
		dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

		self.train_loader = torch.utils.data.DataLoader(dsets["train"], **kwargs)
		self.val_loader = torch.utils.data.DataLoader(dsets["val"], **kwargs)

class LoadSVHN():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': 4,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			datasets.SVHN('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.ToTensor(),
						#Normalization pending
						transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			datasets.SVHN('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   #Normalization pending
						   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
					   ])),
		  **kwargs)

class LoadSTL10():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': 4,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		self.train_loader = torch.utils.data.DataLoader(
			datasets.STL10('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.ToTensor(),
						#Normalization pending
						transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
					   ])),
			 **kwargs)

		self.val_loader = torch.utils.data.DataLoader(
			datasets.STL10('../data', train=False,
			  transform=transforms.Compose([
						   transforms.ToTensor(),
						   #Normalization pending
						   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
					   ])),
		  **kwargs)

class LoadImagenet12():
	def __init__(self, opt):
		kwargs = {
		  'num_workers': 4,
		  'batch_size' : opt.batch_size,
		  'shuffle' : True,
		  'pin_memory': True}

		data_transforms = {
			'train': transforms.Compose([
				transforms.RandomSizedCrop(225),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				utils.ToGray(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]),
			'val': transforms.Compose([
				transforms.Scale(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				utils.ToGray(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		}

		data_dir = '../data/imagenet12'
		dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

		self.train_loader = torch.utils.data.DataLoader(dsets["train"], **kwargs)
		self.val_loader = torch.utils.data.DataLoader(dsets["val"], **kwargs)

class LoadLIP():
	def __init__(self, opt):
		kwargs = {
			'num_workers': 1,
			'batch_size': opt.batch_size,
			'shuffle': True,
			'pin_memory': False}

		data_transforms = {
			'train': transforms.Compose([
				Rescale((321, 321)),
				ToTensor()
			]),
			'val': transforms.Compose([
				Rescale((321, 321)),
				ToTensor()
			])
		}
		data_dir ='../data/LIP/TrainVal_images/TrainVal_images'
		s_dir ='../data/LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations'
		list_dir = '../data/LIP/TrainVal_images/TrainVal_images'

		dsets = {x: LoadLIPDataset(list_file=os.path.join(list_dir, x + '_list'),
                                            root_dir=os.path.join(data_dir, x + '_images'), seg_dir=os.path.join(s_dir, x + '_segmentations'), transform=data_transforms[x]) for x in ['train', 'val']}

		self.train_loader = torch.utils.data.DataLoader(dsets["train"], **kwargs)
		self.val_loader = torch.utils.data.DataLoader(dsets["val"], **kwargs)
