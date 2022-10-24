# %%
import cv2
from transformers import MobileViTFeatureExtractor, MobileViTForSemanticSegmentation
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
# %%

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image= Image.open(r"C:\Users\garla\OneDrive\Desktop\spinning_truck\truckimgs\images0143.jpg")

# %%

# feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/deeplabv3-mobilevit-small")
# model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024")

# %%

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_mask = logits.argmax(1).squeeze(0)

# %%
predicted_mask
# %%
outputs
# %%
image
# %%
ids = np.unique( predicted_mask)
ids
# %%
fig, ax = plt.subplots(1, len(ids), figsize = (20,20))
for i,id in enumerate(ids):
    mask = predicted_mask == id
    ax[i].imshow(mask)
    
    
# %%
ids
# %%

from pathlib import Path
start_folder = Path(r"C:\Users\garla\OneDrive\Desktop\spinning_truck\truckimgs")
end_folder = Path(r"C:\Users\garla\OneDrive\Desktop\spinning_truck\truckimgs_masked")
end_folder.mkdir(exist_ok=True)
jpgs = list(start_folder.glob("*.jpg"))
jpgs[0].name
# %%
for j,jpg in enumerate(jpgs):
    new_name = end_folder /jpg.name
    print(j,jpg,new_name)
    image = Image.open(jpg)
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_mask = logits.argmax(1).squeeze(0)
    car_id = 13
    
    
    carmask = predicted_mask == car_id
    
    new_size =[i*4 for i in carmask.shape]
    print(f"new size = {new_size}")
    carmask = cv2.resize(np.array(carmask, dtype = np.uint8), dsize=new_size, interpolation=cv2.INTER_NEAREST )

    carmask = np.expand_dims(carmask, axis = 2)
    newsize =np.flip( np.array( carmask.shape[0:2]))
    image_np = np.array(image.resize(newsize), dtype=np.uint8)
    image_np.shape, carmask.shape
    car_only = np.where(carmask,image_np, np.ones_like(image_np)*255)
    imagenew = Image.fromarray(car_only)
    imagenew.save(new_name)
# %%
# %%
logits.shape
# %%
predicted_mask.shape
# %%
carmask.shape
# %%
# from skimage.transform import resize
# carmask =np.array( [[1,2],[3,4]])

carmask = carmask.squeeze()
print(carmask.shape)
# carmask2 = np.repeat(carmask,repeats = 1) #.reshape(4,4)
# carmask2 = resize(carmask, [i*2 for i in carmask.shape])
# carmask_cv = cv2.
# carmask2 = cv2.resize(carmask, dsize=(i*2 for i in carmask.shape), interpolation=cv2.INTER_CUBIC)
new_size =[i*4 for i in carmask.shape]
print(f"new size = {new_size}")
carmask2 = cv2.resize(np.array(carmask, dtype = np.uint8), dsize=new_size, interpolation=cv2.INTER_CUBIC)


carmask2.shape
carmask2
# %%
carmask.shape
# %%
