# ✅ Overview
---
This challenge uses a dataset from [Meta Album](https://meta-album.github.io/). All images of this dataset are **512x512 pixels**. 

The [Meta Album](https://meta-album.github.io/) dataset consists of many different domains, such as:
- Animals:
<center>
<img src="https://meta-album.github.io/assets/img/samples/AWA.png" width="300">
</center>

- Insects:
<center>
<img src="https://meta-album.github.io/assets/img/samples/INS.png" width="300">
</center>

- Human actions:
<center>
<img src="https://meta-album.github.io/assets/img/samples/ACT_410.png" width="300">
</center>

- and many more ...

# ✅ Neural Style Transfer

To understand the competition, first let's understand [Neural Style Transfer](https://en.wikipedia.org/wiki/Neural_style_transfer).

[Neural Style Transfer](https://en.wikipedia.org/wiki/Neural_style_transfer) is a process of **"FUSING"** a content image with a style image to get the result stylized image. Here we fuse a bee from the Insects dataset of [Meta Album](https://meta-album.github.io/) with a painting of leaves to generate the result as shown:

<center>
<img src="https://raw.githubusercontent.com/fnachalearn/style-trans-fair/main/images/demo_image_2.png" width="800">
</center>

This challenge uses a Dataset from **[Meta Album](https://meta-album.github.io/).**

# ✅ Task

This challenge's task is image classification, but using [Neural Style Transfered](https://en.wikipedia.org/wiki/Neural_style_transfer) images instead of original one. Here is a demo of an Apoidea:

- The first image on the left is its original image
- The middle image is the painting style we are using for this image
- The image on the right is the result of [Neural Style Transfer](https://en.wikipedia.org/wiki/Neural_style_transfer), and we need to classify this image as an Apoidea

<center>
<img src="https://raw.githubusercontent.com/fnachalearn/style-trans-fair/main/images/demo_image.png" width="800">
</center>

There are approximately 200 tasks in the Development phase and 200 tasks in the Final Phase.
Each task in the dataset consists of a BIASED training set with **90** images, and a BALANCED test set with **180** images, each splited into 9 groups as shown below. The number represents the number of samples per group.
You can see there are 3 groups with 20 examples, 4 times the others, so your model is more likely to be biased towards these group, misclassifying those with Style 2 as Class 2 because of this. Your job is to mitigate this bias as much as possible.

<center>
<img src="https://raw.githubusercontent.com/fnachalearn/style-trans-fair/main/images/data_distribution.jpg" width="800">
</center>

**Caution:** Only the training set is imbalance, whereas the test set is balanced.

The sample dataset in this starting kit conists of **1 task**.
The public data consists of **3 more tasks** for you to play with.
The dataset we use to evaluate your model consists of **200 tasks**.

# ✅ Data description

Each task is composed of:

- A labels.csv file composed of the following columns:

  + "ORIG_CATEGORY_FILE" is the **file used** for the construction of the **category** of the image
  + "CATEGORY" is the **name of the insect category** of the image
  + "ORIG_STYLE_FILE" is the **file used** for the construction of the **style of the image**
  + "STYLE" is the name of the **style used** to build the image
  + "FILE_NAME" is the **name of the image**
  + "label_cat" is the **label** of the image (the category)
- A "content" folder containing the **content** **images** used to build the final images
- A "styles" folder containing the **style images** used to build the final images
- A "stylized" folder containing the **stylized images** obtained using the [Neural Style Transfer](https://en.wikipedia.org/wiki/Neural_style_transfer) technique. These are the images we want to predict the class.