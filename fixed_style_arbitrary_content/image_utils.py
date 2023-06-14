import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]


def show_image(image, title=None):
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()


def load_image(image_name, image_size=256, scale=None):
    img = Image.open(image_name)
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def show_image_from_data_loader(data_loader):
    examples = next(iter(data_loader))

    for label, img in enumerate(examples):
        i = img[0, ...].clone()
        i /= 255
        print(i)
        i = i.permute(dims=[1, 2, 0])
        plt.imshow(i)
        plt.show()
        break


def reformat_tensor_to_image(tensor):
    a = torch.clamp(tensor, min=0, max=255)
    image = a.detach().cpu().numpy()
    image = image.transpose(1, 2, 0).astype(np.uint8)
    return Image.fromarray(image)


def save_debug_image(style_images, content_images, transformed_images, filename, image_size):
    style_images = style_images.detach().cpu().numpy()[0, ...]
    style_images *= 255
    style_images = Image.fromarray(style_images.transpose(1, 2, 0).astype(np.uint8))

    content_images = [reformat_tensor_to_image(x) for x in content_images]
    transformed_images = [reformat_tensor_to_image(x) for x in transformed_images]

    # style_image = Image.fromarray(np.uint8(reformat_tensor_to_image(style_image)))
    # content_images = [reformat_tensor_to_image(x) for x in content_images]
    # transformed_images = [reformat_tensor_to_image(x) for x in transformed_images]

    new_im = Image.new('RGB',
                       (style_images.size[0] + (image_size + 5) * 4, max(style_images.size[1], image_size * 2 + 5)))
    new_im.paste(style_images, (0, 0))

    x = style_images.size[0] + 5
    for i, (a, b) in enumerate(zip(content_images, transformed_images)):
        new_im.paste(a, (x + (image_size + 5) * i, 0))
        new_im.paste(b, (x + (image_size + 5) * i, image_size + 5))

    new_im.save(filename)
