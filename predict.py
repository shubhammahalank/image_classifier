# Imports
import argparse
import torch
from torchvision import transforms

from train import get_classifier, get_model, get_device

from PIL import Image
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='flowers/test/18/image_04256.jpg', help='Path of image to predict')
    parser.add_argument('checkpoint', type=str, default='save_dir/checkpoint.pth', help='path to checkpoint')
    parser.add_argument('-t', '--top_k', type=int, default=7, help='Get top k probabilities')
    parser.add_argument('-c', '--category_names', type=str, default='cat_to_name.json',
                        help='Mapping of categories to real names')
    parser.add_argument('-g', '--gpu', type=str, default='gpu', help='Use GPU for training')
    in_args = parser.parse_args()
    return in_args


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = get_model(checkpoint['architecture'].lower())

    in_units, classifier = get_classifier(checkpoint['hidden_layers'][0],
                                          checkpoint['hidden_layers'][1],
                                          checkpoint['architecture'].lower())

    model.classifier = classifier

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return (model, checkpoint['architecture'], checkpoint['input_size'], checkpoint['output_size'],
            checkpoint['hidden_layers'])


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open image
    img = Image.open(image).convert('RGB')
    # Resize the image
    img.thumbnail(size=(256, 256))
    # Get image dimensions
    width, height = img.size
    # Crop image 224 x 224
    img = img.crop(((width - 224) / 2, (height - 224) / 2, (width + 224) / 2, (height + 224) / 2))

    # Normalize
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    img = img_transform(img)

    # convert to numpy array
    np_img = np.array(img)
    return np_img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)

    ax.set_title(title)

    return ax


def predict(image_path, model, topk, category_name_file, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Get image
    np_image = np.expand_dims(process_image(image_path), axis=0)
    # Convert to tensor
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor).to(device)
    # Get probabiliries
    logps = model.forward(tensor_image).to('cpu')
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    top_p = top_p.detach().cpu().numpy()[0]

    list_p = top_p.tolist()[0]
    list_class = top_class.tolist()[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in list_class]
    top_flowers = [category_name_file[str(lab)] for lab in top_labels]

    return top_p, top_labels, top_flowers


def main():
    in_args = get_arguments()

    checkpoint_path = in_args.checkpoint
    model, architecture, input_size, output_size, hidden_layers = load_checkpoint(checkpoint_path)

    gpu = in_args.gpu
    device = get_device(gpu)
    model.to(device)

    image_path = in_args.input
    category_name = in_args.category_names

    with open(category_name, 'r') as f:
        category_name_file = json.load(f)

    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)

    # Title
    flower_code = image_path.split('/')[2]
    title_ = category_name_file[flower_code]
    # Plot
    img = process_image(image_path)
    imshow(img, ax, title=title_)
    # Get prediction
    top_k = in_args.top_k
    top_p, top_labels, top_flowers = predict(image_path, model, top_k, category_name_file, device)
    # Bar chart
    plt.subplot(2, 1, 2)
    sns.barplot(x=top_p, y=top_flowers)
    print('--------------------------------')
    print(f'Predicted Image Name: {top_flowers[0]} and its probability: {top_p[0]:.2f}\n')
    print(f'Top {top_k} classes and respective probability:')
    for idx in range(len(top_flowers)):
        print(f'{idx+1}: Predicted Image Name: {top_flowers[idx]} and its probability: {top_p[idx]:.2f}')
    print('--------------------------------')
    plt.show()


if __name__ == '__main__':
    main()
