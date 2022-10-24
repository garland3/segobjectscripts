# %%
import matplotlib.pylab as plt
import numpy as np
import io
import requests
from PIL import Image
import torch
import numpy

from transformers import DetrFeatureExtractor, DetrForSegmentation, SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image= Image.open(r"C:\Users\garla\OneDrive\Desktop\spinning_truck\truckimgs\images0143.jpg")
# image = image.resize((1024,1024))
image
# %%


# feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
# model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-101-panoptic', do_resize   = True)
model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-101-panoptic')

# feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")


# %%

# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# forward pass
outputs = model(**inputs)

# use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

# the segmentation is stored in a special-format png
panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
# retrieve the ids corresponding to each mask
panoptic_seg_id = rgb_to_id(panoptic_seg)

# %%
image
# %%
panoptic_seg_id
# %%
panoptic_seg
# %%
panoptic_seg_id.shape
# %%
panoptic_seg.shape
# %%
ids = np.unique( panoptic_seg_id)
ids
# %%
fig, ax = plt.subplots(1, len(ids), figsize = (20,20))
for i,id in enumerate(ids):
    mask = panoptic_seg_id == id
    ax[i].imshow(mask)
    
# %%
category_2_id = {item['category_id']:item['id'] for item in  result['segments_info']}
category_2_id
category_2_id[8]
# %%

car_id =category_2_id[8]
carmask = panoptic_seg_id == car_id
carmask = np.expand_dims(carmask, axis = 2)
carmask.shape

# %%
newsize =np.flip( np.array( carmask.shape[0:2]))
# newsize = np.array( carmask.shape[0:2])

newsize 
# %%


image_np = np.array(image.resize(newsize), dtype=np.uint8)
image_np.shape, carmask.shape
car_only = np.where(carmask,image_np, np.ones_like(image_np)*255)
# %%
car_only.shape
# inputs.keys()
# %%
# inputs.pixel_values.shape
# %%
# feature_extractor.decode()
plt.figure(figsize=(15,15))
plt.imshow(car_only)
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
    processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
    result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

    # the segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
    # retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb_to_id(panoptic_seg)
    category_2_id = {item['category_id']:item['id'] for item in  result['segments_info']}
    car_id =category_2_id[8]
    
    
    carmask = panoptic_seg_id == car_id
    carmask = np.expand_dims(carmask, axis = 2)
    carmask.shape

    newsize =np.flip( np.array( carmask.shape[0:2]))

    image_np = np.array(image.resize(newsize), dtype=np.uint8)
    image_np.shape, carmask.shape
    car_only = np.where(carmask,image_np, np.ones_like(image_np)*255)
    imagenew = Image.fromarray(car_only)
    imagenew.save(new_name)
# %%
result.keys()
# %%
seginfo = result['segments_info']
# %%
seginfo
# %%
