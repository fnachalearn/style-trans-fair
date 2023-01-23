# Important Instructions for the processed version of dataset:
the dataset should be in the following format:

1. A parent directory with your dataset name which consists of the files `labels.csv` and `info.json`
2. preprocessed images (128x128) in a folder named `images` inside the parent directory

<br>
<br>

### `labels.csv`

- FILE_NAME
- CATEGORY
- SUPER_CATEGORY  
- STYLE
- ORIG_CATEOGRY_FILE 
- ORIG_STYLE_FILE


If you choose different column names, specify them in info.json.

<br>
<br>

### `info.json`
 
- dataset_name: `string`
- dataset_description: `string`
- total_categories: `integer`
- total_super_categories: `integer`
- total_styles: `integer`
- number_of_images_per_category: `integer`
- number_of_images_per_style: `integer`
- number_of_images_per_dominant_group: `integer`
- number_of_images_per_non_dominant_group: `integer`
- has_super_categories `boolean`
- image_column_name `string`
- orig_category_image_column_name: `string`
- orig_style_image_column_name: `string`
- category_column_name: `string`
- super_category_column_name: `string`
- style_column_name `string`


<br>
<br>

### Example of insects dataset is given below:
```
task_1/
├── labels.csv
├── info.json
├── stylized/
├──── stylized1.jpg
├──── stylized2.jpg
├──── stylized3.jpg
├── styles/
├──── styles1.jpg
├──── styles2.jpg
├──── styles3.jpg
├── content/
├──── content1.jpg
├──── content2.jpg
├──── content3.jpg


<br>
<br>

### Examples of `labels.csv` files for sample insects dataset:
***With predefined super-category***

```
FILE_NAME,      CATEGORY,   SUPER_CATEGORY
insect1.jpg,    bee,        insect     
insect2.jpg,    wasp,       insect
insect3.jpg,    butterfly,  insect
```

***Without predefined super-category***
```
FILE_NAME,      CATEGORY,   SUPER_CATEGORY
insect1.jpg,    bee,        NAN                   
insect2.jpg,    wasp,       NAN
insect3.jpg,    butterfly,  NAN
```

<br>
<br>

### Examples of `info.json` file for sample insects dataset:

```json
{
    "dataset_name" : "mini_insect_1",
    "dataset_description" : "mini insect example dataset # 1",
    "total_categories" : 4,
    "total_super_categories" : 1,
    "uniform_number_of_images_per_category" : true, 
    "minimum_images_per_category" : 3,
    "median_images_per_category" : 3.0,
    "maximum_images_per_category" : 3,
    "has_super_categories" : true,
    "image_column_name" : "FILE_NAME",
    "category_column_name" : "CATEGORY",
    "super_category_column_name" : "SUPER_CATEGORY",
}
```


<br>
<br>

# Sample Datasets
You will find three sample dataset in this repository

1. `mini_insect_1` : mini dataset with *Super Categories*
2. `mini_insect_2` : mini dataset without *Super Categories*


<br>
<br>


# Check Data Format
To check if your dataset is correctly formatted before running the **Factsheet Script**, use the python script `check_data_format.py`.

You can run the script in the following way 

```
python check_data_format.py --dataset_path './mini_insect_1'
```