import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
from torchvision.utils import save_image


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features


def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 356

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[], std=[])
    ]
)

content_img = load_image("cat.jpg")
style_img = load_image("starry_night.jpg")

model = VGG().to(device).eval()

# generated = torch.randn(original_img, device=device, requires_grad=True)
generated_img = content_img.clone().requires_grad_(True)

total_steps = 6000
learning_rate = 0.001
alpha = torch.tensor(1)
beta = torch.tensor(0.01)
optimizer = optim.Adam([generated_img], lr=learning_rate)

for step in range(total_steps):
    generated_features = model(generated_img)
    content_features = model(content_img)
    style_features = model(style_img)

    style_loss = original_loss = torch.tensor(0)

    for generated_feature, content_feature, style_feature in zip(
            generated_features, content_features, style_features):
        batch_size, channel, height, width = generated_feature.shape
        original_loss += torch.mean((generated_feature - content_feature) ** 2)

        G = generated_feature.view(channel, height * width).mm(
            generated_feature.view(channel, height * width).t())

        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t())

        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print(f"{step} steps,\t loss: {total_loss.item()}")

    if step % 200 == 0:
        save_image(generated_img, f"generated_{step}.png")
