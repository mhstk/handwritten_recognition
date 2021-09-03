  
import numpy as np
import matplotlib.pyplot as plt
import time


# A function to plot images
def show_image(img, title):
    plt.figure(title)
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


# Reading Images Set
def read_image_label(images_file_addr, labels_file_addr):
    start = time.time()
    images_file = open(images_file_addr, 'rb')
    images_file.seek(4)
    num_of_images = int.from_bytes(images_file.read(4), 'big')
    images_file.seek(16)

    labels_file = open(labels_file_addr, 'rb')
    labels_file.seek(8)

    out_set = []
    for n in range(num_of_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(images_file.read(1), 'big') / 256
        label_value = int.from_bytes(labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1
        
        out_set.append((image, label))

    print(f"\ttime spent: {time.time() - start}\n")


    # show_image(out_set[2][0], "Number " + str(np.where(out_set[2][1] == 1)[0][0]) )
    # plt.show()
    return out_set

def corrupted_image(image, shift_number):
    reshaped = image.reshape(28,28)
    shift_image = np.delete(reshaped,np.s_[(28-shift_number):], 1 )
    shift_image = np.concatenate((np.zeros((28,shift_number)), shift_image), axis=1)
    shift_image = shift_image.reshape(784,1)
    return shift_image
    