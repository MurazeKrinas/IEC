import os

path = './Dataset/Images/test_images/'
num = 0
for filename in os.listdir(path):
    sour = path + filename
    des = path + f'test{num}.jpg'
    os.rename(sour, des)
    num += 1
