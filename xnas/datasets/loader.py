import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from xnas.core.config import cfg
from xnas.core.utils import get_project_root
from xnas.datasets.taskonomy_dataset import get_datasets
from xnas.datasets.transforms import *
from xnas.datasets.imagenet16 import ImageNet16
from xnas.datasets.imagenet import ImageFolder
from xnas.datasets import load_ops
from xnas.datasets.transforms import Cutout

SUPPORTED_DATASETS = [
    "cifar10", 
    "cifar100", 
    "svhn", 
    "imagenet16", 
    "mnist", 
    "fashionmnist"
]

# if you use datasets loaded by imagefolder, you can add it here.
IMAGEFOLDER_FORMAT = ["imagenet"]


def construct_loader(
        cutout_length=0,
        use_classes=None,
        transforms=None,
        **kwargs
    ):
    """Construct NAS dataloaders with train&valid subsets."""
    
    split = cfg.LOADER.SPLIT
    name = cfg.LOADER.DATASET
    batch_size = cfg.LOADER.BATCH_SIZE
    datapath = cfg.LOADER.DATAPATH
    
    assert (name in SUPPORTED_DATASETS) or (name in IMAGEFOLDER_FORMAT), "dataset not supported."

    # expand batch_size to support different number during training & validating
    if isinstance(batch_size, int):
        batch_size = [batch_size] * len(split)
    elif batch_size is None:
        batch_size = [256] * len(split)
    assert len(batch_size) == len(split), "lengths of batch_size and split should be same."
    
    # check if randomresized crop is used only in ImageFolder type datasets
    if len(cfg.SEARCH.MULTI_SIZES):
        assert name in IMAGEFOLDER_FORMAT, "RandomResizedCrop can only be used in ImageFolder currently."
    
    if name in SUPPORTED_DATASETS:
        # using training data only.
        train_data, _ = get_data(name, datapath, cutout_length, use_classes=use_classes, transforms=transforms)
        return split_dataloader(train_data, batch_size, split)
    elif name in IMAGEFOLDER_FORMAT:
        assert cfg.LOADER.USE_VAL is False, "do not using VAL dataset."
        aug_type = cfg.LOADER.TRANSFORM
        return ImageFolder( # using path of training data of ImageNet as `datapath`
            datapath, batch_size=batch_size, split=split,
            use_val=False, augment_type=aug_type, **kwargs
        ).generate_data_loader()
    else:
        print("dataset not supported.")
        exit(0) 


def get_data(name, root, cutout_length, download=True, use_classes=None, transforms=None):
    assert name in SUPPORTED_DATASETS, "dataset not support."
    assert cutout_length >= 0, "cutout_length should not be less than zero."
    root = "./data/" + name if not root else os.path.join(root, name)

    if name == "cifar10":
        train_transform, valid_transform = transforms_cifar10(cutout_length) if transforms is None else transforms
        train_data = dset.CIFAR10(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.CIFAR10(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "cifar100":
        train_transform, valid_transform = transforms_cifar100(cutout_length) if transforms is None else transforms
        train_data = dset.CIFAR100(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.CIFAR100(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "svhn":
        train_transform, valid_transform = transforms_svhn(cutout_length) if transforms is None else transforms
        train_data = dset.SVHN(
            root=root, split="train", download=download, transform=train_transform
        )
        test_data = dset.SVHN(
            root=root, split="test", download=download, transform=valid_transform
        )
    elif name == "mnist":
        train_transform, valid_transform = transforms_mnist(cutout_length) if transforms is None else transforms
        train_data = dset.MNIST(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.MNIST(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "fashionmnist":
        train_transform, valid_transform = transforms_mnist(cutout_length) if transforms is None else transforms
        train_data = dset.FashionMNIST(
            root=root, train=True, download=download, transform=train_transform
        )
        test_data = dset.FashionMNIST(
            root=root, train=False, download=download, transform=valid_transform
        )
    elif name == "imagenet16":
        train_transform, valid_transform = transforms_imagenet16() if transforms is None else transforms
        train_data = ImageNet16(
            root=root,
            train=True,
            transform=train_transform,
            use_num_of_class_only=use_classes,
        )
        test_data = ImageNet16(
            root=root,
            train=False,
            transform=valid_transform,
            use_num_of_class_only=use_classes,
        )
        if use_classes == 120 or use_classes is None:   # Use 120 classes by default.
            assert len(train_data) == 151700 and len(test_data) == 6000
        elif use_classes == 150:
            assert len(train_data) == 190272 and len(test_data) == 7500
        elif use_classes == 200:
            assert len(train_data) == 254775 and len(test_data) == 10000
        elif use_classes == 1000:
            assert len(train_data) == 1281167 and len(test_data) == 50000
    else:
        exit(0)
    return train_data, test_data

def get_train_val_loaders(config, mode, indices=None, train_ratio=1):
    """
    Constructs the dataloaders and transforms for training, validation and test data.
    """
    data = config.LOADER.DATAPATH
    dataset = config.LOADER.DATASET
    seed = config.RNG_SEED
    num_workers = config.LOADER.NUM_WORKERS
    config = config.SEARCH if mode == "train" else config.evaluation
    if dataset == "cifar10":
        train_transform, valid_transform = _data_transforms_cifar10(config)
        train_data = dset.CIFAR10(
            root=data, train=True, download=True, transform=train_transform
        )
        test_data = dset.CIFAR10(
            root=data, train=False, download=True, transform=valid_transform
        )
    elif dataset == "cifar100":
        train_transform, valid_transform = _data_transforms_cifar100(config)
        train_data = dset.CIFAR100(
            root=data, train=True, download=True, transform=train_transform
        )
        test_data = dset.CIFAR100(
            root=data, train=False, download=True, transform=valid_transform
        )
    elif dataset == "svhn":
        train_transform, valid_transform = _data_transforms_svhn(config)
        train_data = dset.SVHN(
            root=data, split="train", download=True, transform=train_transform
        )
        test_data = dset.SVHN(
            root=data, split="test", download=True, transform=valid_transform
        )
    elif dataset == "ImageNet16-120":

        train_transform, valid_transform = _data_transforms_ImageNet_16_120(config)
        data_folder = f"{data}/{dataset}"
        train_data = ImageNet16(
            root=data_folder,
            train=True,
            transform=train_transform,
            use_num_of_class_only=120,
        )
        test_data = ImageNet16(
            root=data_folder,
            train=False,
            transform=valid_transform,
            use_num_of_class_only=120,
        )
    elif dataset == 'jigsaw':
        cfg = get_jigsaw_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']
        
    elif dataset == 'class_object':
        cfg = get_class_object_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']
        
    elif dataset == 'class_scene':
        cfg = get_class_scene_configs()

        train_data, val_data, test_data = get_datasets(cfg)

        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']
        
    elif dataset == 'autoencoder':
        cfg = get_autoencoder_configs()
    
        train_data, val_data, test_data = get_datasets(cfg)
        
        train_transform = cfg['train_transform_fn']
        valid_transform = cfg['val_transform_fn']
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    num_train = len(train_data)
    if indices is None:
        indices = list(range(num_train))
        np.random.shuffle(indices)
    split = int(np.floor(config.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split][:int(split*train_ratio)]),
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed+1),
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed),
    )

    test_queue = torch.utils.data.DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed),
    )

    return train_queue, valid_queue, test_queue, train_transform, valid_transform


def get_normal_dataloader(
    name=None,
    train_batch=None,
    cutout_length=0,
    use_classes=None,
    transforms=None,
    **kwargs
):
    name=cfg.LOADER.DATASET if name is None else name
    train_batch=cfg.LOADER.BATCH_SIZE if train_batch is None else train_batch
    name=cfg.LOADER.DATASET
    datapath=cfg.LOADER.DATAPATH
    test_batch=cfg.LOADER.BATCH_SIZE if cfg.TEST.BATCH_SIZE == -1 else cfg.TEST.BATCH_SIZE
    
    assert (name in SUPPORTED_DATASETS) or (name in IMAGEFOLDER_FORMAT), "dataset not supported."
    assert isinstance(train_batch, int), "normal dataloader using single training batch-size, not list."
    # check if randomresized crop is used only in ImageFolder type datasets
    if len(cfg.SEARCH.MULTI_SIZES):
        assert name in IMAGEFOLDER_FORMAT, "RandomResizedCrop can only be used in ImageFolder currently."

    if name in SUPPORTED_DATASETS:
        # get normal dataloaders with train&test subsets.
        train_data, test_data = get_data(name, datapath, cutout_length, use_classes=use_classes, transforms=transforms)
        
        train_loader = data.DataLoader(
            dataset=train_data,
            batch_size=train_batch,
            shuffle=True,
            num_workers=cfg.LOADER.NUM_WORKERS,
            pin_memory=cfg.LOADER.PIN_MEMORY,
        )
        test_loader = data.DataLoader(
            dataset=test_data,
            batch_size=test_batch,
            shuffle=False,
            num_workers=cfg.LOADER.NUM_WORKERS,
            pin_memory=cfg.LOADER.PIN_MEMORY,
        )
        return train_loader, test_loader
    elif name in IMAGEFOLDER_FORMAT:
        assert cfg.LOADER.USE_VAL is True, "getting normal dataloader."
        aug_type = cfg.LOADER.TRANSFORM
        return ImageFolder( # using path of training data of ImageNet as `datapath`
            datapath, batch_size=[train_batch, test_batch],
            use_val=True, augment_type=aug_type, **kwargs
        ).generate_data_loader()

def split_dataloader(data_, batch_size, split):
    assert 0 not in split, "illegal split list with zero."
    assert sum(split) == 1, "summation of split should be one."
    num_data = len(data_)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    portion = [int(sum(split[:i]) * num_data) for i in range(len(split) + 1)]

    return [
        data.DataLoader(
            dataset=data_,
            batch_size=batch_size[i - 1],
            sampler=data.sampler.SubsetRandomSampler(
                indices[portion[i - 1] : portion[i]]
            ),
            num_workers=cfg.LOADER.NUM_WORKERS,
            pin_memory=cfg.LOADER.PIN_MEMORY,
        )
        for i in range(1, len(portion))
    ]


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_svhn(args):
    SVHN_MEAN = [0.4377, 0.4438, 0.4728]
    SVHN_STD = [0.1980, 0.2010, 0.1970]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_ImageNet_16_120(args):
    IMAGENET16_MEAN = [x / 255 for x in [122.68, 116.66, 104.01]]
    IMAGENET16_STD = [x / 255 for x in [63.22, 61.26, 65.09]]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(16, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET16_MEAN, IMAGENET16_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET16_MEAN, IMAGENET16_STD),
        ]
    )
    return train_transform, valid_transform


def get_jigsaw_configs():
    cfg = {}
    
    cfg['task_name'] = 'jigsaw'

    cfg['input_dim'] = (255, 255)
    cfg['target_num_channels'] = 9
    
    cfg['dataset_dir'] = os.path.join(get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = os.path.join(get_project_root(), "data", "final5K_splits")
    
    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'
    
    cfg['target_dim'] = 1000
    cfg['target_load_fn'] = load_ops.random_jigsaw_permutation
    cfg['target_load_kwargs'] = {'classes': cfg['target_dim']}
    
    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.RandomGrayscale(0.3),
        load_ops.MakeJigsawPuzzle(classes=cfg['target_dim'], mode='max', tile_dim=(64, 64), centercrop=0.9, norm=False, totensor=True),
    ])
    
    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.RandomGrayscale(0.3),
        load_ops.MakeJigsawPuzzle(classes=cfg['target_dim'], mode='max', tile_dim=(64, 64), centercrop=0.9, norm=False, totensor=True),
    ])
    
    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.RandomGrayscale(0.3),
        load_ops.MakeJigsawPuzzle(classes=cfg['target_dim'], mode='max', tile_dim=(64, 64), centercrop=0.9, norm=False, totensor=True),
    ])
    return cfg


def get_class_object_configs():
    cfg = {}
    
    cfg['task_name'] = 'class_object'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3
    
    cfg['dataset_dir'] = os.path.join(get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = os.path.join(get_project_root(), "data", "final5K_splits")
    
    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'
    
    cfg['target_dim'] = 75
    
    cfg['target_load_fn'] = load_ops.load_class_object_logits
    
    cfg['target_load_kwargs'] = {'selected': True if cfg['target_dim'] < 1000 else False, 'final5k': True if cfg['data_split_dir'].split('/')[-1] == 'final5k' else False}
    
    cfg['demo_kwargs'] = {'selected': True if cfg['target_dim'] < 1000 else False, 'final5k': True if cfg['data_split_dir'].split('/')[-1] == 'final5k' else False}
    
    cfg['normal_params'] = {'mean': [0.5224, 0.5222, 0.5221], 'std': [0.2234, 0.2235, 0.2236]}
    
    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    
    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    
    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
       load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg


def get_class_scene_configs():
    cfg = {}
    
    cfg['task_name'] = 'class_scene'

    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3
    
    cfg['dataset_dir'] = os.path.join(get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = os.path.join(get_project_root(), "data", "final5K_splits")
    
    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'
    
    cfg['target_dim'] = 47
    
    cfg['target_load_fn'] = load_ops.load_class_scene_logits
    
    cfg['target_load_kwargs'] = {'selected': True if cfg['target_dim'] < 365 else False, 'final5k': True if cfg['data_split_dir'].split('/')[-1] == 'final5k' else False}
    
    cfg['demo_kwargs'] = {'selected': True if cfg['target_dim'] < 365 else False, 'final5k': True if cfg['data_split_dir'].split('/')[-1] == 'final5k' else False}
    
    cfg['normal_params'] = {'mean': [0.5224, 0.5222, 0.5221], 'std': [0.2234, 0.2235, 0.2236]}
    
    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    
    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    
    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
       load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg


def get_autoencoder_configs():
    
    cfg = {}
    
    cfg['task_name'] = 'autoencoder'
    
    cfg['input_dim'] = (256, 256)
    cfg['input_num_channels'] = 3
    
    cfg['target_dim'] = (256, 256)
    cfg['target_channel'] = 3

    cfg['dataset_dir'] = os.path.join(get_project_root(), "data", "taskonomydata_mini")
    cfg['data_split_dir'] = os.path.join(get_project_root(), "data", "final5K_splits")
   
    cfg['train_filenames'] = 'train_filenames_final5k.json'
    cfg['val_filenames'] = 'val_filenames_final5k.json'
    cfg['test_filenames'] = 'test_filenames_final5k.json'
    
    cfg['target_load_fn'] = load_ops.load_raw_img_label
    cfg['target_load_kwargs'] = {}
    
    cfg['normal_params'] = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
    
    cfg['train_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.RandomHorizontalFlip(0.5),
        load_ops.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    
    cfg['val_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    
    cfg['test_transform_fn'] = load_ops.Compose(cfg['task_name'], [
        load_ops.ToPILImage(),
        load_ops.Resize(list(cfg['input_dim'])),
        load_ops.ToTensor(),
        load_ops.Normalize(**cfg['normal_params']),
    ])
    return cfg
