import random
import skeltorch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms


class MNISTClassifierData(skeltorch.Data):
    train_indexes = None
    validation_indexes = None
    transforms = None

    def create(self, data_path):
        self.load_datasets(data_path)

        # Create a list containing the indexes of the default MNIST Train split
        train_set_len = len(self.datasets['train'])
        train_set_indexes = list(range(train_set_len))
        random.shuffle(train_set_indexes)

        # Create a validation split using the percentage of data specified in the configuration file
        val_split = self.experiment.configuration.get('data', 'val_split')
        self.train_indexes = train_set_indexes[:round((1 - val_split) * len(train_set_indexes))]
        self.validation_indexes = train_set_indexes[round((1 - val_split) * len(train_set_indexes)):]

    def load_datasets(self, data_path):
        self._load_transforms()
        self.datasets['train'] = torchvision.datasets.MNIST(
            data_path, train=True, transform=self.transforms, download=True
        )
        self.datasets['validation'] = self.datasets['train']
        self.datasets['test'] = torchvision.datasets.MNIST(
            data_path, train=False, transform=self.transforms, download=True
        )

    def load_loaders(self, data_path, num_workers):
        self.loaders['train'] = torch.utils.data.DataLoader(
            dataset=self.datasets['train'],
            sampler=torch.utils.data.SubsetRandomSampler(self.train_indexes),
            batch_size=self.experiment.configuration.get('training', 'batch_size'),
            num_workers=num_workers
        )
        self.loaders['validation'] = torch.utils.data.DataLoader(
            dataset=self.datasets['validation'],
            sampler=torch.utils.data.SubsetRandomSampler(self.validation_indexes),
            batch_size=self.experiment.configuration.get('training', 'batch_size'),
            num_workers=num_workers
        )
        self.loaders['test'] = torch.utils.data.DataLoader(
            dataset=self.datasets['test'],
            shuffle=True,
            batch_size=self.experiment.configuration.get('training', 'batch_size'),
            num_workers=num_workers
        )

    def _load_transforms(self):
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
