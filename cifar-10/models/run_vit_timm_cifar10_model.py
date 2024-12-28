import timm
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# load model
model = timm.create_model("vit_base_patch16_384", pretrained=False)
model.head = nn.Linear(model.head.in_features, 10)


ckpt_path = "./vision-transformers-cifar10/vit_timm-4-ckpt.t7"
checkpoint = torch.load(ckpt_path)

# model was trained using torch.nn.DataParallel.
# strip the "module." prefix from the keys before loading the state dictionary into the model.
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[len('module.'):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# Remove 'module.' prefix from the checkpoint state dictionary
state_dict = checkpoint['model']
state_dict = remove_module_prefix(state_dict)

model.load_state_dict(state_dict)

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load dataset
size = 384

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=8)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# evaluation mode
model.eval()
test_loss = 0
correct = 0
total = 0
model.to(device)

correct = 0
total = 0
with torch.no_grad():
    loop = tqdm(testloader, total=len(testloader))
    for (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        loop.set_description(f"{correct} / {total} Accuracy {100.*correct/total:.2f}%")

print(f"correct: {predicted.eq(targets).sum().item() } out of {targets.size(0)}, accuracy {100.*correct/total}")

def predict (inputs):
    # model.eval()
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, predicted = outputs.max(1)
    return predicted