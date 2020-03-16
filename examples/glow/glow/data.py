import skeltorch
import torch.utils.data
import torchvision.transforms


class GlowData(skeltorch.Data):
    transforms = None

    def create(self, data_path):
        pass

    def load_datasets(self, data_path):
        self._load_transforms()
        self.datasets['train'] = torchvision.datasets.CIFAR10(
            data_path, train=True, transform=self.transforms, download=True
        )
        self.datasets['validation'] = torchvision.datasets.CIFAR10(
            data_path, train=False, transform=self.transforms, download=True
        )

    def load_loaders(self, data_path, num_workers):
        self.loaders['train'] = torch.utils.data.DataLoader(
            dataset=self.datasets['train'],
            shuffle=True,
            batch_size=self.experiment.configuration.get('training', 'batch_size'),
            num_workers=num_workers
        )
        self.loaders['validation'] = torch.utils.data.DataLoader(
            dataset=self.datasets['validation'],
            shuffle=True,
            batch_size=self.experiment.configuration.get('training', 'batch_size'),
            num_workers=num_workers
        )

    def _load_transforms(self):
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
