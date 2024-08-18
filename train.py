# Imports
import os
import argparse
import torch
from collections import OrderedDict
from torch import nn, optim, utils
from torchvision import datasets, transforms, models


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default='flowers',
                        help='Path of data directory')
    parser.add_argument('-sd', '--save_dir', type=str, default='checkpoint.pth',
                        help='Set directory to save checkpoints')
    parser.add_argument('-a', '--arch', type=str, default='MNASNet',
                        help='Choose architecture (vgg / alexnet / MNASNet)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-hu1', '--hidden_units1', type=int, default=640, help='Number of units in first hidden layer')
    parser.add_argument('-hu2', '--hidden_units2', type=int, default=320, help='Number of units in second hidden layer')
    parser.add_argument('-e', '--epochs', type=int, default=18, help='Number of epochs')
    parser.add_argument('-g', '--gpu', type=str, default='gpu', help='Use GPU for training')
    in_args = parser.parse_args()
    return in_args


def transform_load(train_dir, valid_dir, test_dir):
    # Transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=train_transforms)

    # Dataloaders using the image datasets and the trainforms
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    data_transforms = {
        'train': train_transforms,
        'valid': valid_transforms,
        'test': test_transforms
    }

    image_datasets = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }

    dataloaders = {
        'train': trainloader,
        'valid': validloader,
        'test': testloader
    }

    return data_transforms, image_datasets, dataloaders


def get_model(architecture):
    if architecture == 'vgg':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif architecture == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    elif architecture == 'mnasnet':
        model = models.mnasnet1_3(weights=models.MNASNet1_3_Weights.DEFAULT)
    else:
        model = models.mnasnet1_3(weights=models.MNASNet1_3_Weights.DEFAULT)

    print(f'{architecture} architecture is being used')
    # Freeze model parameters to avoid backdrop
    for param in model.parameters():
        param.requires_grad = False

    return model


def get_classifier(hidden_units1, hidden_units2, architecture):
    if architecture == 'vgg':
        in_units = 25088
    elif architecture == 'alexnet':
        in_units = 9216
    elif architecture == 'mnasnet':
        in_units = 1280
    else:
        in_units = 1280

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_units, hidden_units1)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units1, hidden_units2)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(hidden_units2, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    return in_units, classifier


def get_device(gpu):
    if gpu == 'cpu':
        device = 'cpu'
    else:
        if torch.cuda.is_available():
            device = 'cuda:0'
            print('GPU device CUDA is being used')
        elif torch.backends.mps.is_available():
            device = 'mps'
            print('GPU device MPS is being used')
        else:
            device = 'cpu'
            print('GPU device is not available')
    return device


def train_and_validate(model, device, epochs, optimizer, criterion, dataloaders):
    # Training
    running_loss = 0
    training_accuracy = 0
    print_every = 50
    for epoch in range(epochs):
        steps = 0
        trainloader = dataloaders['train']
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            training_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            if steps % print_every == 0:
                validation_loss = 0
                validation_accuracy = 0
                model.eval()

                with torch.no_grad():
                    validloader = dataloaders['valid']
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f'Epoch {epoch + 1}/{epochs} | '
                      f'Train loss: {running_loss / print_every:.2f} | '
                      f'Train accuracy: {(training_accuracy / print_every) * 100:.2f}% | '
                      f'Validation loss: {validation_loss / len(validloader):.2f} | '
                      f'Validation accuracy: {(validation_accuracy / len(validloader)) * 100:.2f}%')
                running_loss = 0
                training_accuracy = 0
                model.train()
    return model


def model_test(model, device, criterion, dataloaders):
    model.eval()
    test_accuracy = 0
    testloader = dataloaders['test']
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)

        loss = criterion(logps, labels)

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f'Test accuracy: {(test_accuracy / len(testloader)) * 100:.2f}%')


def save_checkpoint(model_, save_dir, image_datasets, in_args, in_units):
    model_.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'architecture': in_args.arch,
                  'input_size': in_units,
                  'output_size': 102,
                  'hidden_layers': [in_args.hidden_units1, in_args.hidden_units2],
                  'state_dict': model_.state_dict(),
                  'class_to_idx': model_.class_to_idx,
                  'epochs': in_args.epochs,
                  'optimizer_dict': model_.state_dict(),
                  }

    # Create save_dir if it doesn't exist
    if not os.path.exists(save_dir):
        print('directory does not exist')
        os.makedirs(save_dir)
        print('directory created')
    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')
    print(f'checkpoint successfully saved at: {save_dir}')


def main():
    in_args = get_arguments()

    data_dir = in_args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms, image_datasets, dataloaders = transform_load(train_dir, valid_dir, test_dir)

    architecture = in_args.arch
    model = get_model(architecture.lower())

    hidden_units1 = in_args.hidden_units1
    hidden_units2 = in_args.hidden_units2
    in_units, classifier = get_classifier(hidden_units1, hidden_units2, architecture)

    model.classifier = classifier
    criterion = nn.NLLLoss()
    learning_rate = in_args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    gpu = in_args.gpu
    device = get_device(gpu)
    model.to(device)

    epochs = in_args.epochs
    learned_model = train_and_validate(model, device, epochs, optimizer, criterion, dataloaders)

    model_test(learned_model, device, criterion, dataloaders)

    save_dir = in_args.save_dir
    save_checkpoint(learned_model, save_dir, image_datasets, in_args, in_units)


if __name__ == '__main__':
    main()
