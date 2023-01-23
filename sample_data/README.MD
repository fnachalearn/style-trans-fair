# Important Instructions for the processed version of dataset:

the dataset should be in the following format:

1. A parent directory with your dataset name which consists of the files `labels.csv` and `info.json`
2. preprocessed images (512x512) in a folder named `content`, `styles`, `stylized` inside the parent directory

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

### Examples of `info.json` file for sample insects dataset:

```json
{
    "dataset_name" : "STYLE-TRANSFER-FAIR", 
    "dataset_description" : "",
    "total_categories" : 0,
    "total_super_categories" : 0,
    "total_styles" : 3,
    "number_of_images_per_category" : 0,
    "number_of_images_per_style" : 0,
    "number_of_images_per_dominant_group" : 0,
    "has_super_categories" : false,
    "image_column_name" : "FILE_NAME",
    "orig_category_image_column_name" : "ORIG_CATEGORY_FILE",
    "orig_style_image_column_name" : "ORIG_STYLE_FILE",
    "category_column_name" : "CATEGORY",
    "super_category_column_name" : "SUPER_CATEGORY",
    "style_column_name" : "STYLE"
}
```

<br>
<br>
