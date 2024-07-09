import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tkinter import Tk, filedialog

# Initialize the Blip model and processor
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base").to("cuda")

# Create a Tkinter root window (it will not be shown)
root = Tk()
root.withdraw()

# Open a file dialog for image selection
img_path = filedialog.askopenfilename(title="Select an image file", filetypes=[
                                      ("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

# Check if a file was selected
if not img_path:
    print("No file selected. Exiting.")
    exit()

# Load the selected image
raw_image = Image.open(img_path).convert('RGB')

# Conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# Unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
