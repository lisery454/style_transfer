import time

import torch
from torch import optim
from torchvision import transforms

from fixed_style_arbitrary_content import image_utils, data_utils, model_utils, math_utils

batch_size = 4
image_size = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

style_path = "origin/rain-princess.jpg"
dataset_path = r'E:\Data\imagenet-mini\train'

# [1, 3, 256, 256]
style_img = image_utils.load_image(style_path, image_size)
style_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda i: i.mul(255))
])
style_img = style_transform(style_img)

style_img = style_img.repeat(batch_size, 1, 1, 1).to(device)

dataset = data_utils.get_data_set(dataset_path, image_size)
data_loader = data_utils.get_data_loader_from_file(dataset_path, image_size, batch_size)

vgg = model_utils.VGG().to(device).eval()

style_features = vgg(math_utils.normalize_batch(style_img))
style_grams = [math_utils.gram_matrix(y) for y in style_features]

transform_net = model_utils.TransformNet().to(device)

epoch_count = 1
verbose_batch_count = 100
style_weight = 10e10
content_weight = 10e5
learn_rate = 1e-3
optimizer = optim.Adam(transform_net.parameters(), learn_rate)
mse_loss = torch.nn.MSELoss()
batch_count = len(data_loader)

for e in range(epoch_count):
    print('Epoch: {}'.format(e + 1))
    transform_net.train()
    agg_content_loss = 0.
    agg_style_loss = 0.
    count = 0
    for batch_id, (x, _) in enumerate(data_loader):

        n_batch = len(x)
        optimizer.zero_grad()

        x = x.to(device)
        y = transform_net(x)

        if batch_id % verbose_batch_count == 0:
            image_utils.save_debug_image(style_img, x, y, f"debug/s_{e}_{batch_id}.jpg", image_size)

        n_y = math_utils.normalize_batch(y)
        n_x = math_utils.normalize_batch(x)

        features_y = vgg(n_y)
        features_x = vgg(n_x)

        content_loss = content_weight * mse_loss(features_y[2], features_x[2])

        style_loss = 0.
        for ft_y, gm_s in zip(features_y, style_grams):
            gm_y = math_utils.gram_matrix(ft_y)
            style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
        style_loss *= style_weight

        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()

        agg_content_loss += content_loss.item()
        agg_style_loss += style_loss.item()

        if batch_id % verbose_batch_count == 0:
            mesg = f"{time.ctime()}\tEpoch {e + 1}:\t[{count}/{len(dataset)}]\t" \
                   f"content: {agg_content_loss / (batch_id + 1):.6f}\t" \
                   f"style: {agg_style_loss / (batch_id + 1):.6f}\t" \
                   f"total: {(agg_content_loss + agg_style_loss) / (batch_id + 1):.6f}"
            print(mesg)

        count += n_batch

        if count >= 12000:
            break

    transform_net.eval().cpu()
    torch.save(transform_net.state_dict(), 'transform_net.pth')
