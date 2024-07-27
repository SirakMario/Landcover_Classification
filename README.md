# Land Cover Classification from Aerial Image Using U-net in Poland


# Introduction
Land cover refers to the earth’s surface features, including elements such as water, soil, vegetation, and their diverse sub-categories. Accurate and up-to-date land cover classification is important for effective environmental monitoring, urban planning, and sustainable resource management. In Poland, rapid urbanization, agricultural activities, and climate change have increased the need for precise land cover information. Traditional methods of land cover classification, which depends on manual interpretation of satellite images and field surveys, are time consuming, labor-intensive, and sensitive to human errors. The introduction of U-net architecture, deep learning (DL), technology presents a transformative opportunity to enhance land cover classification processes. Using high-resolution aerial imagery and DL algorithms, it is possible to automate the classification process, significantly improve accuracy, efficiency, and scalability .

# Study area
![Study area](https://github.com/SirakMario/Landcover_Classification/blob/main/assets/study_area.png)
# Dataset
The dataset (Fig. 2) is, from LandCover .ai [1] (Land Cover from Aerial Imagery), designed for the automatic mapping of land covers such as buildings, water, roads, and woodland from aerial images of Poland
• Raster Images and Masks are 3-channel and 1-channel GeoTiffs respectively
• 33 orthophotos with 25 cm per pixel resolution (~9000x9500 px)
• 8 orthophotos with 50 cm per pixel resolution (~4200x4700 px)
• Total area: 216.27 km2
• Class: Unlabeled (0), Building (1), woodland (2), water (3), road (4)
![Dataset used](https://github.com/SirakMario/Landcover_Classification/blob/main/assets/datasets.png)
# Objective
1. Develop a U-Net model that accurately classifies land cover
2. To  evaluate  and  compare  the  performance  of  U-Net  model  with  and without data augmentation. The aim is to show the improved effectiveness of  augmented  trained  data  in  generating  reliable  land  cover  data  for environmental monitoring and urban planning purposes in Poland.
# Methodology
![Methodology](https://github.com/SirakMario/Landcover_Classification/blob/main/assets/Methodology.PNG)
# Results
Fig4.  presents  the  land  cover  classification  results  obtained  from  the  U-net  (ResNet32),  a convolutional  neural  network  architecture,  applied  to  aerial  images  of  Poland.  The  input  masks consist of five classes: building, woodland, water, road, and unlabeled. This study employed the Segmentation  Models  library  in  Python,  which  is  based  on  the  Keras  (Tensorflow)  framework. Image  segmentation  was  performed  in  two  approaches:  using  only  the  original  images  and using  augmented  images.  The  augmentations  included  horizontal  and  vertical  flips,  with  a method  to  handle  pixels  generated  outside  the  image  boundaries  during  transformation.  The performance  difference  between  these  two  approaches  is  significant.  Evaluation  metrics, Intersection  over  Union  (IOU),  demonstrate  that  classification  with  augmented  data outperforms classification without augmentation, as shown in Table 1.
![Result table](https://github.com/SirakMario/Landcover_Classification/blob/main/assets/Table.png)
![result](https://github.com/SirakMario/Landcover_Classification/blob/main/assets/results.png)
# Conclusion
Land cover classification is important in sustainable urban and regional
planning.
• The model shows robust performance, outperforming traditional
classification methods.


