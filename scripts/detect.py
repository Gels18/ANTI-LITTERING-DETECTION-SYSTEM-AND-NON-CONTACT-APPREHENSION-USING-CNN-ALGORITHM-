import os
from ultralytics import YOLO

model = YOLO(r"C:/Users/Administrator/Desktop/Litter-Detect/models/best.pt")# Specific na file location ng na train na model
image_path = r"C:\Users/Administrator/Desktop/Litter-Detect/images/basura.jpg"# location ng image na i ttry as sample 
                      
results = model(image_path, save=True, project="C:/Users/Administrator/Desktop/Litter-Detect/output") #dito naman po massave yung detected po na basura

for result in results: # yung result mapupunta yung pic sa same folder pero nasa output folder
    print(result)


