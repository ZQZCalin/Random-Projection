import os
import torch
import torch.nn as nn

from tqdm import tqdm

# ==> Pre-processing dataset

class ProcessedCIFAR100(torch.utils.data.Dataset):
    def __init__(
        self, root: str, train: bool, model: nn.Module,
        dataloader: torch.utils.data.DataLoader
    ):
        super(ProcessedCIFAR100, self).__init__()
        self.dataset = self.process_cifar100(root, train, model, dataloader)

    @torch.no_grad()
    def process_cifar100(self,
        root: str, train: bool, model: nn.Module, dataloader: torch.utils.data.DataLoader):
        '''
        Pre-process CIFAR100 dataset and save processed features 
        and labels as hashmaps.
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        mode = 'train' if train else 'test'
        file_path = f'{root}/{mode}.pt'

        # check existence for processed training set
        if os.path.exists(file_path):
            print(f'Preprocessed {mode}ing data already existed.')
            return torch.load(file_path)

        else:
            print(f'Processing {mode}ing data..')

            if not os.path.isdir(root):
                os.mkdir(root)

            # store feature and label in hashmap
            dataset = {}
            idx = 0

            pbar = tqdm(enumerate(dataloader))
            for batch, (inputs, labels) in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                for i in range(outputs.size(0)):
                    dataset[idx] = (outputs[i], labels[i])
                    idx += 1
                pbar.set_description(f'batch {batch+1}')

            # save to local data
            torch.save(dataset, file_path)
            return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# testing code
# if __name__ == '__main__':
    # for b, (inputs, labels) in enumerate(trainloader):
    #     outputs = resnet50_no_last(inputs)
    #     print(outputs.size(), outputs)
    #     if b >= 0:
    #         break
    # print(len(trainloader))

    # trainset_processed = ProcessedCIFAR100(train=True)
    # testset_processed = ProcessedCIFAR100(train=False)
    
    # pass