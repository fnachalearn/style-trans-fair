# ✅ Overview
---
This challenge uses a dataset from [Meta Album](https://meta-album.github.io/). All images of this dataset are resized into **512x512 pixels**. The following image is an example of a stylized image of an insect.

<img src="https://github.com/fnachalearn/public_style-trans-fair/blob/main/dataset-cover.jpg?raw=true" width="600">

# ✅ Data description
---

This challenge is composed of the two following phases:

- The **development phase**:
	+ 5 tasks
- The **final phase**:
	+ 5 other tasks

Each task is composed of:

- A labels.csv file composed of the following columns:
	+ "ORIG_CATEGORY_FILE" is the **file used** for the construction of the **category** of the image
	+ "CATEGORY" is the **name of the insect category** of the image
	+ "ORIG_STYLE_FILE" is the **file used** for the construction of the **style of the image**
	+ "STYLE" is the name of the **style used** to build the image
	+ "FILE_NAME" is the **name of the image**
	+ "label_cat" is the **label** of the image (the category)
	

- A "content" folder containing the **insect images** used to build the final images
	
	
- A "styles" folder containing the **style images** used to build the final images
	
	
- A "stylized" folder containing the **final stylized images** obtained with the "style-transfert" technique from an insect image and a style image. It is those images we want to predict the class.


Each task in this dataset consits of **360 labeled images** of insects splitted into **9 groups** distributed with **3 styles and 3 classes**.

The 9 groups of images are imbalanced, resulting in bias in the dataset: each class is more represented in one of the 3 styles in order to fool machine learning models which will correlate classes to their corresponding style.

<img src="https://drive.google.com/uc?id=1S0z1E724-INIYxSrql9iWZaLmvY0NJgn" height="300">