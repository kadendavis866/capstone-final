import random
from multiprocessing.spawn import freeze_support
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model import build_model

def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    dataroot = 'balancer/centered/sam90'
    IMAGE_SIZE = 128
    num_channels = 3
    batch_size = 128
    num_epochs = 10
    num_gpus = 1
    workers = 2
    device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus > 0) else "cpu")

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(10),
                                   transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    classifier = build_model().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters())

    ckpt_path = Path("best_models_synth/sam90u18.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_inputs = 0
        for i, data in enumerate(dataloader):
            classifier.zero_grad()
            inputs = data[0].to(device)
            labels = data[1].type(torch.FloatTensor).to(device)
            output = classifier(inputs).view(-1)
            err = criterion(output, labels)
            running_loss += err * inputs.size(0)
            num_inputs += inputs.size(0)
            err.backward()
            optimizer.step()
        running_loss = running_loss / num_inputs
        print(epoch + 1, running_loss)

        # Save only when improved
        if running_loss < best_loss:
            best_loss = running_loss
            torch.save(
                {
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "num_channels": num_channels,
                },
                ckpt_path,
            )
            print(f"saved new best to {ckpt_path} (loss={best_loss:.6f})")


if __name__ == '__main__':
    freeze_support()
    main()