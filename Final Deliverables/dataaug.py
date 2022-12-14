from keras . preprocessing. image import ImageDataGenerator
import cv2
from os import listdir
import time

# Nicely formatted time string to make a note of how much time it takes for augmentation
def hms_string (sec_elapsed) :
    h=int(sec_elapsed / (60 * 60) )
    m=int ((sec_elapsed % (60 * 60) ) / 60)
    s=sec_elapsed%60
    return f"{h}: {m}:{round(s, 1)}"

def augment_data (file_dir, n_generated_samples, save_to_dir) :
    """Arguments:
file_dir: A string representing the directory where images that we want to augment are found.
i-generated samples. A string representing the number of generated samples using the giv
save_to_dir: A string representing the directory in which the generated images will be saved."""


#from keras . preprocessing. image import ImageDataGenerator
#from os import listdir
    data_gen = ImageDataGenerator( 
        rotation_range=30,

        width_shift_range=0.1,
        height_shift_range=0.15,
        shear_range=0-25,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode= 'nearest' ,
        brightness_range= (0.5, 1.2))
    for filename in listdir(file_dir) :
    # load the image
        image = cv2. imread(file_dir + '/' + filename)
        # reshape the image
        image = image . reshape ( (1,)+image . shape)
        # prefix of the names for the generated sampels.

        save_prefix = 'aug_' + filename [:-4]
        #generate 'n_generated_samples' sample images
        i=0
        for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_to_dir,save_prefix=save_prefix, save_format='jpg' ) :
            i+=1
            if i>n_generated_samples :
                break
start_time=time.time()
augumented_data_path='C:/Users/Beni PC/Desktop/main-project/augumented_data/'
augment_data(file_dir='C:/Users/Beni PC/Desktop/main-project/Digital_naturalist_data/Bird/Great Indian Bustard Bird',n_generated_samples=8,save_to_dir=augumented_data_path+'Bird/GIB_AUG')
augment_data(file_dir='C:/Users/Beni PC/Desktop/main-project/Digital_naturalist_data/Bird/Spoon Billed Sandpiper Bird',n_generated_samples=8,save_to_dir=augumented_data_path+'Bird/SPS_AUG')

augment_data(file_dir='C:/Users/Beni PC/Desktop/main-project/Digital_naturalist_data/Flower/Corpse Flower',n_generated_samples=0,save_to_dir=augumented_data_path+'Flower/Corpse_AUG')
augment_data(file_dir='C:/Users/Beni PC/Desktop/main-project/Digital_naturalist_data/Flower/Lady Slipper Orchid Flower',n_generated_samples=0,save_to_dir=augumented_data_path+'Flower/LS_Orchid_AUG')

augment_data(file_dir='C:/Users/Beni PC/Desktop/main-project/Digital_naturalist_data/Mammal/Pangolin Mammal',n_generated_samples=0,save_to_dir=augumented_data_path+'Mammal/Pangolin_AUG')
augment_data(file_dir='C:/Users/Beni PC/Desktop/main-project/Digital_naturalist_data/Mammal/Senenca White Deer Mammal',n_generated_samples=0,save_to_dir=augumented_data_path+'Mammal/SW_Deer_AUG')