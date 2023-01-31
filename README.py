#!/usr/bin/env python
# coding: utf-8

# In[2]:


# RIGHT NOW, PRIVATE REPOS DOES NOT SUPPORT IMAGES IN MARKDOWN
from IPython.display import Image
Image(filename="images/dark_logo.jpg", width=200)
# WE CHANGED IT IN THE STARTING KIT UPLOADED ON CODABENCH


# <div style="background:#FFFFFF">
# <!-- <img src="../images/dataset-cover.jpg" width=150 ALIGN="left" style='margin-right:10px; border-style: solid; border-width: 2px;' alt='logo'> -->
# <h1>Starting Kit - Style-Trans-Fair </h1>
# <p>
# This starting kit will guide you step by step and will walk you through the data statistics and<br>
# examples. This will give you a clear idea of what this challenge is about and how you can<br>
# proceed further to solve the challenge.
# </p>
# 
# <br><br>
# <hr style='background-color: #D3D3D3; height: 1px; border: 0;'>
# <p>
# This code was tested with Python 3.8.5 |Anaconda custom (64-bit)| (default, Dec 23 2020, 21:19:02) (https://anaconda.org/)<br>
# </p>
# <hr style='background-color: #D3D3D3; height: 1px; border: 0;'>
#     
# <p>
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". The CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. IN NO EVENT SHALL AUTHORS AND ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 
# </p>
# 
# <hr style='background-color: #D3D3D3; height: 1px; border: 0;'>
#     <p>
# This challenge was organized by <b>Team Fairness National Assembly</b> of <b><a style='color:#4D6605;' href='http://www.chalearn.org/'>ChaLearn</a></b>  at <b><a style='color: #62023C;' href='https://www.universite-paris-saclay.fr/'>Université Paris Saclay</a></b>
# </p>
# </div>
# 
# <hr style='background-color: #D3D3D3; height: 1px; border: 0;'>

# ***
# # Introduction
# 
# This challenge uses a Dataset from ***Meta Album***. There are 5 tasks in the Development phase and 5 tasks in the Final Phase.
# Each task in the dataset consists of **360** images splited into 9 groups:
# <!-- 
# <p>
# <img src="../images/task_distribution.png">
# </p> -->

# In[2]:


# RIGHT NOW, PRIVATE REPOS DOES NOT SUPPORT IMAGES IN MARKDOWN
Image(filename="images/task_distribution.png") 
# WE CHANGED IT IN THE STARTING KIT UPLOADED ON CODABENCH


# 
# 
# where the training set is biased in Style, and the test set is balanced.
# 
# The sample dataset in this starting kit conists of **360** images of the Insect classification task.
#     
# This challenge is about creating a Machine Learning model and train it with the data provided to classify the images into the mentioned 3 Classes without being biased towards Styles.
# 
# 
# **References and credits:**  
#  - Meta Album (https://meta-album.github.io/)  
#  - Université Paris Saclay (https://www.universite-paris-saclay.fr/)  
#  - ChaLearn (http://www.chalearn.org/)  
# ***

# ### Install required packages

# In[5]:


get_ipython().system('pip install -r requirements.txt')


# ## Attention: 
# 
# The starting baseline is using a convolution neural network built with tensorflow. For the participants who want to use the GPU, you will need to install additonal dependencies depending on the operating system you use. Here are some guidelines depending on your OS preference: 
# 
# 
# ***
# <b>Windows</b> : https://machinelearningprojects.net/install-tensorflow-with-cuda-and-cudnn/
# 
# <b>Linux</b> : https://www.tensorflow.org/install/pip
# 
# <b>MacOS</b> : https://medium.com/mlearning-ai/install-tensorflow-on-mac-m1-m2-with-gpu-support-c404c6cfb580
# ***

# ### Imports

# In[6]:


# import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Image
from PIL import Image
from sklearn.decomposition import PCA
import os


# In[7]:


model_dir = 'sample_code_submission/' # Change the model to a better one once you have one!
result_dir = 'sample_result_submission/' 
problem_dir = 'ingestion_program/'  
score_dir = 'scoring_program/'
from sys import path; path.append(model_dir); path.append(problem_dir); path.append(score_dir); 
get_ipython().run_line_magic('matplotlib', 'inline')
# Uncomment the next lines to auto-reload libraries (this causes some problem with pickles in Python 3)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import seaborn as sns; sns.set()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[8]:


get_ipython().run_line_magic('reload_ext', 'autoreload')


# ***
# # Step 1: Exploratory data analysis
# We provide `sample_data` with the starting kit, but to prepare your submission, you must fetch the `public_data` from the challenge website and point to it.
# 
# The data used for this challenge has images resized into 512x512 pixels.

# In[9]:


data_name = 'style_trans_fair_challenge' # DO NOT CHANGE
data_dir = 'sample_data' # Change it to point to the directory with public_data
image_dir = os.path.join(data_dir, 'stylized')


# ### Load Data

# In[10]:


from data_io import read_data
data, meta_data = read_data(os.path.join(data_dir, 'task1'), random_state=42)


# ### Data Statistics

# In[11]:


print("Categories/Classes : ", np.unique(data['train_labels']))
print("Styles : ", np.unique(data['train_styles']))
print("Train Images:", len(data['train_images']))
print("Test Images:", len(data['test_images']))


# ## Visualization

# In[12]:


def print_style_transfer_samples(row):
    f, ax = plt.subplots(1, 3)
    f.set_figheight(5)
    f.set_figwidth(15)
    ax[0].imshow(Image.open(os.path.join(data_dir, 'task1', 'content', row['ORIG_CATEGORY_FILE'].iloc[0])))
    ax[0].set_title('Original')
    ax[0].set_axis_off()
    ax[1].imshow(Image.open(os.path.join(data_dir, 'task1', 'styles', row['ORIG_STYLE_FILE'].iloc[0])))
    ax[1].set_title('Style')
    ax[1].set_axis_off()
    ax[2].imshow(Image.open(os.path.join(data_dir, 'task1', 'stylized', row['FILE_NAME'].iloc[0])))
    ax[2].set_title('Stylized')
    ax[2].set_axis_off()


# In[13]:


print_style_transfer_samples(data['train_df'].sample(1))


# ### Distribution of Classes/Labels

# #### Train

# In[14]:


grouped = data['train_df'].groupby(["CATEGORY", "STYLE"]).size().reset_index(name="COUNT")
grouped


# In[15]:


# Create the heatmap visualization for category and style
sns.heatmap(grouped.pivot("CATEGORY", "STYLE", "COUNT"), cmap="crest", annot=True, linewidths=1, linecolor='black')
plt.title('Heatmap visualization for category and style')
plt.xticks(rotation=45)
# Show the plot
plt.show()


# #### Test

# In[16]:


grouped = data['test_df'].groupby(["CATEGORY", "STYLE"]).size().reset_index(name="COUNT")
grouped


# In[17]:


# Create the heatmap visualization for category and style
sns.heatmap(grouped.pivot("CATEGORY", "STYLE", "COUNT"), cmap="crest", annot=True, linewidths=1, linecolor='black')
plt.title('Heatmap visualization for category and style')
plt.xticks(rotation=45)
# Show the plot
plt.show()


# ### Further visualization

# For further exploratory data analysis, we analyze the RGB color model and plot the graph. We get the mean for each channel of each image, and save it in the columns of a dataframe.

# In[18]:


rgb = pd.DataFrame(
    np.zeros((len(data['train_df']), 5)), columns=list("RGB") + ["category", "style"]
)
for i in range(len(data['train_labels'])):
    img = data['train_images'][i]
    rgb.iloc[i, :3] = img.mean(axis=(0,1))
    rgb.iloc[i, 3] = data['train_labels'][i]
    rgb.iloc[i, 4] = data['train_styles'][i]
rgb.head()


# We also normalize each color vector, so that we only care about the direction of the color vector and not about its magnitude.

# In[19]:


pca = PCA(2)
rgb[list("rgb")] = rgb[list("RGB")] / np.sqrt(
    np.sum(np.square(rgb[list("RGB")]), axis=1).values.reshape(-1, 1)
) 

transformed_rgb = pca.fit_transform(rgb[list("rgb")]) 
rgb[list("xy")] = transformed_rgb 
red, green, blue = pca.transform(np.identity(3))


# In[20]:


plot_data = rgb
plt.figure(figsize=(10, 10))
sns.scatterplot(x="x", y="y", hue="category", data=plot_data, palette="Set1")
plt.arrow(0, 0, *blue, color="blue")
plt.arrow(0, 0, *green, color="green")
plt.arrow(0, 0, *red, color="red")
plt.title("RGB distribution")
plt.xlabel("First PC")
plt.ylabel("Second PC")
plt.legend()
plt.show()


# Next, we analyze the brightness of the images per category. We wanted to know which category has lower/higher brightness level so instead of using a simple mean for each RGB channel, we used a weighted mean that takes into account the perceived brightness by humans:
# - Red: 21.26% 
# - Green: 71.52% 
# - Blue: 7.22%

# In[21]:


BRIGHTNESS_VECTOR = np.array([0.2126, 0.7152, 0.0722])
rgb["brightness"] = (
    rgb[list("RGB")] @ BRIGHTNESS_VECTOR / 255
)
rgb.head()


# Using the boxplot we could easily identify the minimum, maximum, interquartile range, median and the outliers. Here we ordered it in respect to their lower quartile (Q1) to make it more representative.

# In[22]:


plot_data = rgb[["category", "brightness"]].copy()
order = plot_data.groupby("category")["brightness"].quantile(0.25).sort_values().index.tolist()
plt.figure(figsize=(8, 8))
sns.boxplot(y="category", x="brightness", data=plot_data, order=order, palette="Set1")
plt.title("Luminance (perceived brightness)") 
plt.show()


# In[23]:


ax = pd.DataFrame(rgb.groupby("category")["brightness"].mean()).plot.barh()
plt.xlabel("Brightness level")
plt.title("Plot of the brightness level")
plt.show()


# From the graph above we can conclude that the category Elateroidea has the highest value in term of brightness level

# ***
# # Step 2: Building a predictive model
# 

# ## Training a predictive model
# We provide an example of predictive model in the `sample_code_submission/` directory. 
# The model will perform the following steps:
# - Resize the image to (64,64,3)
# - Flatten the image to a vector of size 64\*64\*3 = 12288
# - Train a simple Support Vector Machine to classify
#   
# You should change this model and use a better one to get a good score for the challenge
# 

# In[24]:


from data_io import write
from model import model


# <div style="background:#FFF">
# an instance of the model (run the constructor) and attempt to reload a previously saved version from `sample_code_submission/`:
# </div>

# In[25]:


myModel = model()
trained_model_name = model_dir + data_name
# Uncomment the next line to re-load an already trained model
#myModel = myModel.load(trained_model_name) 


# <div style="background:#FFF">
#     Train the model (unless you reloaded a trained model) and make predictions. 
# </div>

# In[26]:


X_TRAIN = data["train_images"]
STYLE_TRAIN = data["train_styles"]
Y_TRAIN = data["train_labels_num"]
X_TEST = data["test_images"]
STYLE_TEST = data["test_styles"]
Y_TEST = data["test_labels_num"]


# In[27]:


if not(myModel.is_trained):
    myModel.fit(X_TRAIN, Y_TRAIN)                     

Y_hat_train = myModel.predict(X_TRAIN) # Optional, not really needed to test on taining examples
Y_hat_test = myModel.predict(X_TEST)


# **Save the trained model** (will be ready to reload next time around) and save the prediction results. <br>

# In[28]:


myModel.save(trained_model_name)    


# **IMPORTANT:** if you save the trained model, it will be bundled with your sample code submission. Therefore your model will NOT be retrained on the challenge platform. Remove the pickle from the submission if you want the model to be retrained on the platform.
# 
# **REQUIRED:** Trained model is required in the submission to codabench. 

# In[29]:


result_name = result_dir + data_name
from data_io import write
write(result_name + '_train.predict', Y_hat_train)
write(result_name + '_test.predict', Y_hat_test)

get_ipython().system('ls $result_name*')


# ## Scoring the results
# ### Load the challenge metric
# 
# **The metric chosen for your challenge** is identified in the "metric.txt" file found in the `scoring_program/` directory.
# <br> 
# The function "get_metric" searches first for a metric having that name in my_metric.py, then in libscores.py, then in sklearn.metric.
# 
# Currently, we are using geometric_mean_accuracy_metric, which is calculated by:
# - Split the data in 9 groups, where images in each group have the same style and category
# - Calculate the accuracy in each group
# - Calculate the Geometric Mean of those 9 accuracies
# 

# In[30]:


from libscores import get_metric
metric_name, scoring_function = get_metric()
print('Using scoring metric:', metric_name)
# Uncomment the next line to display the code of the scoring metric
#??scoring_function


# ## Training performance

# In[31]:


print('Training score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_TRAIN, Y_hat_train, STYLE_TRAIN))
print('Ideal score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_TRAIN, Y_TRAIN, STYLE_TRAIN))

print('Test score for the', metric_name, 'metric = %5.4f' % scoring_function(Y_TEST, Y_hat_test, STYLE_TEST))


# You can add here other scores and result visualization.

# ## Confusion matrix

# In[32]:


from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(Y_TRAIN, Y_hat_train)


# In[33]:


confusion_matrix


# In[34]:


df_cm = pd.DataFrame(confusion_matrix, range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1]))
sns.heatmap(df_cm, annot=True)
plt.title("Train - Confustion Matrix")
plt.show()


# ***
# # Step 3: Making a submission
# 
# ## Unit testing
# 
# It is <b><span style="color:red">important that you test your submission files before submitting them</span></b>. All you have to do to make a submission is modify the file <code>model.py</code> in the <code>sample_code_submission/</code> directory, then run this test to make sure everything works fine. This is the actual program that will be run on the server to test your submission. 
# <br>
# Keep the sample code simple.<br>
# 
# <code>python3</code> is required for this step

# In[35]:


# !source activate python3; 
get_ipython().system('python3 $problem_dir/ingestion.py $data_dir $result_dir $problem_dir $model_dir')


# ### Test scoring program

# In[36]:


scoring_output_dir = 'scoring_output'
# !source activate deeplearning; 
get_ipython().system('python3 $score_dir/score.py $data_dir $result_dir $scoring_output_dir')


# # Prepare the submission

# In[37]:


import datetime 
from data_io import zipdir
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
sample_code_submission = 'sample_code_submission_' + the_date + '.zip'
# sample_result_submission = 'sample_result_submission_' + the_date + '.zip'
zipdir(sample_code_submission, model_dir)
# zipdir(sample_result_submission, result_dir)
print("Submit this file to codalab:\n" + sample_code_submission)


# In[ ]:




