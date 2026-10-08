import torch
from torchvision.datasets import MNIST
from transformers import AutoImageProcessor, ResNetForImageClassification

MODEL_ID = "microsoft/resnet-50"
BATCH_SIZE = 32

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = ResNetForImageClassification.from_pretrained(MODEL_ID)
model.to(device)
model.eval()

dataset = MNIST(root="data", train=False, download=True)

correct = 0
total = 0

with torch.inference_mode():
    for start in range(0, len(dataset), BATCH_SIZE):
        batch = [
            dataset[i]
            for i in range(start, min(start + BATCH_SIZE, len(dataset)))
        ]

        images = [image.convert("RGB") for image, _ in batch]
        labels = torch.tensor(
            [label for _, label in batch],
            device=device,
        )

        inputs = processor(images=images, return_tensors="pt").to(device)
        predictions = model(**inputs).logits.argmax(dim=-1)

        correct += (predictions == labels).sum().item()
        total += len(batch)

accuracy = correct / total
print(f"Model: {MODEL_ID}")
print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
