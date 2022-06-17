from regex import F
import torch
from torch import optim, nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

import torchvision

from YTDataset import YTDataset

def train(batch_size, num_classes=600, train_split=0.7, epochs=10, lr=1e-4, device="cpu", max_imgs_per_class=10):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Create transformer
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((230, 230)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create a dataset
    dataset = YTDataset('../data/train', max_imgs_per_class=5)

    # Split into train and val
    train_dataset, val_dataset = train_test_split(dataset, test_size=1-train_split)

    # Split val and test
    val_dataset, test_dataset = train_test_split(val_dataset, test_size=0.5)

    # Create a dataloader for train and val
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Create a model
    model = torchvision.models.convnext_base(pretrained=True).cuda(device=device)
    model.classifier[2] = nn.Linear(in_features=model.classifier[2].in_features, out_features=num_classes)

    # Create an optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Create a triplet loss function using cosine similarity
    loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y))

    # Train the model
    for epoch in range(epochs):
        for i, (label, anchor, positive, negative) in enumerate(train_loader):
            # Move to GPU
            anchor = anchor.cuda(device=device)
            positive = positive.cuda(device=device)
            negative = negative.cuda(device=device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            anchor = model(anchor)
            positive = model(positive)
            negative = model(negative)

            loss_value = loss(anchor, positive, negative)

            # Backward pass
            loss_value.backward()

            # Update the weights
            optimizer.step()

            # Print the loss
            if i % 100 == 0:
                print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, loss_value.item()))

        # Running validation to get accuracy
        with torch.no_grad():
            total_loss = 0
            correct = 0
            total = 0
            for i, (label, anchor, positive, negative) in enumerate(val_loader):
                # Move to GPU
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

                # Forward pass
                anchor = model(anchor)
                positive = model(positive)
                negative = model(negative)

                loss_value = loss(anchor, positive, negative)
                total_loss += loss_value.item()

                # Get the predicted class using a softmax function
                anchor_pred = F.softmax(anchor, dim=1)
                
                _, predicted = torch.max(anchor_pred.data, 1)

                # Get the correct class
                total += label.size(0)
                correct += (predicted == label).sum().item()

            # Print the accuracy
            print('Epoch: {}, Val_Accuracy: {}, Test_Loss: {}'.format(epoch, correct / total, total_loss / total))

    # Running at the end test to get accuracy
    with torch.no_grad():
        total_loss = 0
        correct = 0
        total = 0

        for i, (label, anchor, positive, negative) in enumerate(test_loader):
            # Move to GPU
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()

            # Forward pass
            anchor = model(anchor)
            positive = model(positive)
            negative = model(negative)

            loss_value = loss(anchor, positive, negative)
            total_loss += loss_value.item()

            # Get the predicted class using a softmax function
            anchor_pred = F.softmax(anchor, dim=1)
            
            _, predicted = torch.max(anchor_pred.data, 1)

            # Get the correct class
            total += label.size(0)
            correct += (predicted == label).sum().item()

        # Print the accuracy
        print('Test_Accuracy: {}, Test_Loss: {}'.format(correct / total, total_loss / total))

if __name__ == '__main__':
    model = torchvision.models.convnext_small(pretrained=True)
    print(model)
