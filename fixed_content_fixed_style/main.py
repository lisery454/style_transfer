import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
import time


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


image_size = 256


def load_image(image_name):
    image = Image.open(image_name)
    loader = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    image = loader(image).unsqueeze(0)
    return image.to(device)


def save_image(image, file_path):
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = image.clone().squeeze()
    img = denorm(img).clamp_(0, 1)
    torchvision.utils.save_image(img, file_path)


def gram_matrix(y):
    (_, ch, h, w) = y.shape
    features = y.view(ch, w * h)
    features_t = features.t()
    gram = features.mm(features_t)
    return gram


def get_content_loss(generated, content):
    return torch.mean((generated[2] - content[2]) ** 2, dtype=torch.float64).to(device)


def get_style_loss(generated, style):
    loss = torch.tensor(0, dtype=torch.float64).to(device)

    for g, s in zip(generated, style):
        _, c, h, w = g.shape

        G = gram_matrix(g)
        S = gram_matrix(s)

        loss += torch.mean(((G - S) ** 2) / (c * h * w), dtype=torch.float64).to(device)

    return loss


def get_device():
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using : {d}")
    return d


device = get_device()

content_img = load_image("origin/cat.jpg")
style_img = load_image("origin/starry_night.jpg")

print("load img over")

model = VGG().to(device).eval()

# generated = torch.randn(original_img, device=device, requires_grad=True)
generated_img = content_img.clone().requires_grad_(True)

total_steps = 300
# learning_rate = 0.003
alpha = torch.tensor(1, dtype=torch.float64).to(device)
beta = torch.tensor(100, dtype=torch.float64).to(device)
# optimizer = optim.Adam([generated_img], lr=learning_rate, betas=(0.5, 0.999))
optimizer = optim.LBFGS([generated_img])

print("start generation")
start_time = time.time()

step = [0]
while step[0] <= total_steps:
    def f():
        optimizer.zero_grad()

        generated_features = model(generated_img)
        content_features = model(content_img)
        style_features = model(style_img)

        style_loss = get_style_loss(generated_features, style_features)
        content_loss = get_content_loss(generated_features, content_features)
        total_loss = alpha * content_loss + beta * style_loss

        total_loss.backward()
        step[0] += 1
        if step[0] % 50 == 0:
            print(
                f"{step[0]} steps:\n"
                f"\t total_loss: {total_loss.item()},"
                f"\t content_loss: {content_loss.item()},"
                f"\t style_loss: {style_loss.item()}")

        return total_loss

    optimizer.step(f)

end_time = time.time()
print("end generation")

save_image(generated_img, f"generated_imgs/generated.png")

print(f"Run Time : {end_time - start_time} s")
