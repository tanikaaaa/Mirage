import torch
import random
import numpy as np
import torch.utils.data
import logging
from collections import defaultdict

from torchvision import datasets, transforms
from tqdm import tqdm

logger = logging.getLogger("logger")


class MSPDataloader():

    def __init__(self, params):
        self.params = params

        if self.params['load_data_from_pkl'] == True:
            pre_cached_data = torch.load(self.params['pre_cache_data_path'])
            self.train_dataloader = pre_cached_data['train_dataset']
            self.test_dataloader = pre_cached_data['test_dataset']
        else:
            self.load_dataset()

    def load_dataset(self):

        # ---------------- CIFAR transforms ----------------
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        # ---------------- GTSRB transforms ----------------
        transform_gtsrb = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        transform_gtsrb_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        # ---------------- Dataset selection ----------------
        if self.params["dataset"].upper() == "CIFAR10":
            self.train_dataset = datasets.CIFAR10(
                self.params['data_dir'],
                train=True,
                download=True,
                transform=transform_train
            )
            self.test_dataset = datasets.CIFAR10(
                self.params['data_dir'],
                train=False,
                download=True,
                transform=transform_test
            )

        elif self.params["dataset"].upper() == "CIFAR100":
            self.train_dataset = datasets.CIFAR100(
                self.params['data_dir'],
                train=True,
                download=True,
                transform=transform_train
            )
            self.test_dataset = datasets.CIFAR100(
                self.params['data_dir'],
                train=False,
                download=True,
                transform=transform_test
            )

        elif self.params["dataset"].upper() == "GTSRB":
            base_dataset = datasets.GTSRB(
                self.params['data_dir'],
                split="train",
                download=True,
                transform=transform_gtsrb
            )

            # Efficient duplication (instead of list * 3)
            from torch.utils.data import ConcatDataset
            self.train_dataset = ConcatDataset([base_dataset] * 3)

            self.test_dataset = datasets.GTSRB(
                self.params['data_dir'],
                split="test",
                download=True,
                transform=transform_gtsrb_test
            )

        else:
            raise ValueError("Unsupported dataset")

        # ---------------- Federated split ----------------
        indices_per_participant = self.sample_dirichlet_train_data(
            self.params['no_of_total_participants'],
            alpha=self.params['dirichlet_alpha']
        )

        from torch.utils.data import Subset

        train_loaders = []

        for pos, indices in tqdm(indices_per_participant.items()):
            tmp_subset = Subset(self.train_dataset, indices)

            train_loader = torch.utils.data.DataLoader(
                tmp_subset,
                batch_size=self.params["train_batch_size"],
                shuffle=True,
                drop_last=True,
                num_workers=2,
                pin_memory=True
            )

            train_loaders.append(train_loader)   # ✅ IMPORTANT

        self.train_dataloader = train_loaders

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.params["test_batch_size"],
            shuffle=False,
            drop_last=True,
            num_workers=4,
            pin_memory=True
        )

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):

        cifar_classes = {}

        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]

        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])

            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha])
            )

            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))

                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)

                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list