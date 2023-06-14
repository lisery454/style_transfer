import time

import torch
from torchvision import transforms

from fixed_style_arbitrary_content import image_utils, model_utils


def stylize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_image = image_utils.load_image('origin/cat.jpg')
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = model_utils.TransformNet().to(device)
        state_dict = torch.load('transform_net_rain_princess_2.pth')
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        start_time = time.time()
        for i in range(10000):
            style_model(content_image)
        end_time = time.time()

        print(f"use time : {(end_time - start_time) / 10000}s")
        output = style_model(content_image).cpu()
        image_utils.save_image('output/cat_rain_princess_2.jpg', output[0])


stylize()
