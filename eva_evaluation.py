import torch
from torch import nn

# Import Kaiko EVA components
from eva import core
from eva.vision import datasets, transforms

# Import DINOv2 helpers
from dinov2.hub.backbones import dinov2_vitl14_reg  # ViT-L/14 with registers:contentReference[oaicite:3]{index=3}
from dinov2.utils.utils import load_pretrained_weights  # convenient loader:contentReference[oaicite:4]{index=4}

# -----------------------------------------------------------------------------
# 1. Build the DINOv2 backbone and load your self‑supervised checkpoint
# -----------------------------------------------------------------------------
# Create a ViT-L/14 with 4 register tokens, matching your TCGA training setup.
# We set pretrained=False so that it doesn’t download Meta’s weights.
backbone = dinov2_vitl14_reg(
    pretrained=False,  # we will load our own weights
    img_size=518,      # match the crop size used during DINOv2 training
)

# Load the teacher checkpoint.  If your .pth file contains a dict with a
# specific key (e.g. "teacher") then pass checkpoint_key="teacher"; otherwise
# leave checkpoint_key=None.  The loader strips 'module.' and 'backbone.'
# prefixes before loading:contentReference[oaicite:5]{index=5}.
checkpoint_path = (
    "/home/paul/pathologyDino/logs/vitl16_pathology_30ep_pretrained/checkpoints/iter_009999/teacher.pth"
    # "/home/paul/pathologyDino/logs/vitl16_pathology_30ep/checkpoints/iter_014999/teacher.pth"
)
load_pretrained_weights(backbone, checkpoint_path, checkpoint_key=None)

# Freeze the backbone so only the head learns during EVA’s training
for p in backbone.parameters():
    p.requires_grad = False

# Determine the backbone’s embedding dimension (1 024 for ViT‑L/14:contentReference[oaicite:6]{index=6})
embed_dim = backbone.embed_dim

# -----------------------------------------------------------------------------
# 2. Define the classification head and wrap everything in EVA’s HeadModule
# -----------------------------------------------------------------------------
head = nn.Linear(embed_dim, 4)  # BACH dataset has 4 categories

trainer = core.Trainer(max_steps=100)

model = core.HeadModule(
    backbone=backbone,
    head=head,
    criterion=nn.CrossEntropyLoss(),
)

# -----------------------------------------------------------------------------
# 3. Prepare the data.  We use the standard BACH dataset and resize/crop
#     images to 518×518 so they fit the ViT‑L/14 input size.
# -----------------------------------------------------------------------------
# If ResizeAndCrop accepts a size argument, pass size=518.  Otherwise, adjust
# the transform to match your model’s input resolution (e.g. use Resize and
# CenterCrop separately).
data_transforms = transforms.ResizeAndCrop(size=518)

data = core.DataModule(
    datasets=core.DatasetsSchema(
        train=datasets.BACH(
            root="data/bach",
            split="train",
            download=True,
            transforms=data_transforms,
        ),
        val=datasets.BACH(
            root="data/bach",
            split="val",
            download=True,
            transforms=data_transforms,
        ),
    ),
    dataloaders=core.DataloadersSchema(
        train=core.DataLoader(batch_size=32, shuffle=True),
        val=core.DataLoader(batch_size=32),
    ),
)

# -----------------------------------------------------------------------------
# 4. Run the evaluation
# -----------------------------------------------------------------------------
pipeline = core.Interface()
val_scores, test_scores = pipeline.fit(trainer, model=model, data=data)

print("Validation scores:")
for k, v in val_scores.items():
    print(f"  {k:20s}: {v:.4f}")

print("\nTest scores:")
for k, v in test_scores.items():
    print(f"  {k:20s}: {v:.4f}")