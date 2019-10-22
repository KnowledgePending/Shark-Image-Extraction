# Shark-Image-Extraction
🦈🔬Underwater Shark Image Extraction Without Machine Learning  
  
<a href="https://colab.research.google.com/drive/1eSH45KKT83ONK2_-Ix3j42Qw0VHj0fWF"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
### Prerequisites
```
pip3 install opencv-contrib-python numpy
```
## Algorithm:
1. Create multiple images from a combination of different channels and color spaces
2. Create an image with a detailed view of the shark in grayscale and color 
3. Create a binary mask that can identify the shark and part of the image we don't want 
4. Create a 2nd binary mask that can identify the shark and a different part of the image we don't want 
5. Combine and clean the two masks to get a rough outline of the shark
6. Fill the outline/edges so that it can actually be used as a mask
7. Use the mask to extract only the shark from the detailed image
8. Crop the image to only contain the shark using the mask as a guide
9. Replace the background color with white
10. Display both a Grayscale and Color version of the shark  
