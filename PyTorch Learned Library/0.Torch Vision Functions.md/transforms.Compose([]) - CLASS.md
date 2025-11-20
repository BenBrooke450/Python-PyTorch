
# **Full Summary of `torchvision.transforms`**

`torchvision.transforms` is used to **preprocess**, **augment**, and **normalize** image data.
There are **four major categories**:

1. **Conversion transforms** (PIL ↔ Tensor ↔ other formats)
2. **Geometric transforms** (resize, crop, flip, affine, rotate …)
3. **Color transforms** (brightness, contrast, hue …)
4. **Advanced augmentation policies** (AutoAugment, RandAugment, TrivialAugmentWide …)

I’ll cover **all transforms**, with **definitions & examples**.

---

# 1. **Conversion Transforms**

## **1.1 ToTensor**

Converts a PIL Image / numpy array → PyTorch tensor

* Scales pixel values to **[0, 1]**

```python
transforms.ToTensor()
```

---

## **1.2 PILToTensor**

Similar but NO scaling. Pixels stay 0–255.

```python
transforms.PILToTensor()
```

---

## **1.3 ToPILImage**

Convert tensor → PIL image

```python
transforms.ToPILImage()
```

---

## **1.4 ConvertImageDtype**

Changes tensor dtype (useful after ToTensor)

```python
transforms.ConvertImageDtype(torch.float32)
```

---

# 2. **Geometric Transforms**

## **2.1 Resize**

Resizes image to given size.

```python
transforms.Resize((224, 224))
```

---

## **2.2 CenterCrop**

Crop the center of an image.

```python
transforms.CenterCrop(224)
```

---

## **2.3 RandomCrop**

Crop randomly.

```python
transforms.RandomCrop(128)
```

---

## **2.4 FiveCrop & TenCrop**

Produce 5 or 10 crops.

```python
transforms.FiveCrop(224)
```

---

## **2.5 RandomResizedCrop**

Crop + resize randomly (used in ImageNet & ViT training).

```python
transforms.RandomResizedCrop(224, scale=(0.5, 1.0))
```

---

## **2.6 RandomHorizontalFlip / RandomVerticalFlip**

Flip randomly.

```python
transforms.RandomHorizontalFlip(p=0.5)
```

---

## **2.7 RandomRotation**

Rotate randomly.

```python
transforms.RandomRotation(30)
```

---

## **2.8 RandomAffine**

General affine transform: rotate, translate, shear, scale.

```python
transforms.RandomAffine(
    degrees=40,
    translate=(0.1, 0.1),
    scale=(0.8, 1.2),
    shear=15
)
```

---

## **2.9 Pad**

Add padding.

```python
transforms.Pad(10)
```

---

## **2.10 RandomPerspective**

Warp image with random perspective.

```python
transforms.RandomPerspective(distortion_scale=0.6)
```

---

## **2.11 ElasticTransform**

Elastic distortions (used in handwritten data augmentation)

```python
transforms.ElasticTransform(alpha=50.0)
```

---

## **2.12 GaussianBlur**

Apply Gaussian blurring.

```python
transforms.GaussianBlur(kernel_size=(5, 9))
```

---

## **2.13 RandomErasing**

Applied during training to erase random box regions (like CutOut).

```python
transforms.RandomErasing()
```

---

# 3. **Color & Pixel Transforms**

## **3.1 Normalize**

Apply channel-wise normalization:

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
```

---

## **3.2 ColorJitter**

Randomly change brightness, contrast, saturation, hue.

```python
transforms.ColorJitter(brightness=0.2, contrast=0.3)
```

---

## **3.3 Grayscale**

Convert image to grayscale.

```python
transforms.Grayscale(num_output_channels=1)
```

---

## **3.4 RandomGrayscale**

Randomly convert to grayscale.

```python
transforms.RandomGrayscale(p=0.1)
```

---

## **3.5 RandomInvert**

Invert colors.

```python
transforms.RandomInvert()
```

---

## **3.6 RandomPosterize**

Reduce bit depth.

```python
transforms.RandomPosterize(bits=3)
```

---

## **3.7 RandomSolarize**

Solarize (invert above threshold).

```python
transforms.RandomSolarize(threshold=128)
```

---

## **3.8 RandomAdjustSharpness**

Sharpen or unsharpen.

```python
transforms.RandomAdjustSharpness(sharpness_factor=2)
```

---

## **3.9 RandomAutocontrast**

Auto contrast.

```python
transforms.RandomAutocontrast()
```

---

## **3.10 RandomEqualize**

Equalize histogram.

```python
transforms.RandomEqualize()
```

---

# 4. **Advanced Augmentation Policy Transforms**

## **4.1 AutoAugment**

Uses learned augmentation policies (from Google AutoAugment).

```python
transforms.AutoAugment()
```

Policies include

* CIFAR10
* ImageNet
* SVHN

---

## **4.2 RandAugment**

Applies N random transformations at varying magnitudes.

```python
transforms.RandAugment(num_ops=2, magnitude=9)
```

---

## ⭐ **4.3 TrivialAugmentWide (YOU ASKED ABOUT THIS)**

Applies **one random augmentation at random magnitude**.
No tuning, simple, powerful.

```python
transforms.TrivialAugmentWide()
```

---

## **4.4 AugMix**

Creates mixed augmentations + convex combinations for stability.

```python
transforms.AugMix()
```

---

## **4.5 RandomChoice**

Choose one transform from a list.

```python
transforms.RandomChoice([transforms.RandomRotation(10),
                         transforms.ColorJitter()])
```

---

## **4.6 RandomApply**

Apply a list of transforms with probability p.

```python
transforms.RandomApply(
    [transforms.ColorJitter(), transforms.GaussianBlur(3)],
    p=0.3
)
```

---

## **4.7 RandomOrder**

Apply transforms in *random order*.

```python
transforms.RandomOrder([
    transforms.RandomRotation(20),
    transforms.ColorJitter()
])
```

---

# 5. **Transforms for Video / Tensor-level**

Torchvision v2 includes transforms that operate on:

* Batched Tensors `(B, C, H, W)`
* Videos `(T, C, H, W)`

Examples:

```python
transforms.v2.RandomResizedCrop(224)
transforms.v2.Normalize(mean, std)
transforms.v2.ColorJitter()
```

These are identical in concept but **work on tensors instead of PIL images**.

---

# 6. **Functional API (torchvision.transforms.functional)**

If you want direct function calls:

```python
from torchvision.transforms import functional as F

img = F.rotate(img, 45)
img = F.adjust_brightness(img, 1.2)
img = F.perspective(img, startpoints, endpoints)
```

Good for writing **custom transforms**.

---

# 7. **Custom Transforms**

You can write your own:

```python
class MyTransform:
    def __call__(self, img):
        return F.invert(img)
```

---

# Final Notes

You now have the **complete and detailed list** of:

* Basic transforms
* Geometric transforms
* Color transforms
* Tensor & video transforms
* Advanced augmentation (AutoAugment, RandAugment, AugMix, TrivialAugmentWide)
* Functional transforms
* Custom transforms


