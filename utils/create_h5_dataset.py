from tflearn.data_utils import build_hdf5_image_dataset
import h5py

new_train = "/home/meteo/zihao.chen/fine-grained-classifi/train_test/train_data"
new_val = "/home/meteo/zihao.chen/fine-grained-classifi/train_test/validation_data.txt"
new_test = "/home/meteo/zihao.chen/fine-grained-classifi/train_test/test_data.txt"

upload_test = '/home/meteo/zihao.chen/fine-grained-classifi/TEST_FIlE'
# image_shape option can be set to different values to create images of different sizes
# build_hdf5_image_dataset(new_val, image_shape=(448, 448), mode='file', output_path='new_val_448.h5', categorical_labels=True, normalize=False)
# print 'Done creating new_val_448.h5'
# build_hdf5_image_dataset(new_test, image_shape=(448, 448), mode='file', output_path='new_test_448.h5', categorical_labels=True, normalize=False)
# print 'Done creating new_test_448bc .h5'
# build_hdf5_image_dataset(new_train+'_%d.txt'%(0), image_shape=(224, 224), mode='file', output_path='new_train_224_%d.h5'% (0), categorical_labels=True, normalize=False)

for index in range(11):
    build_hdf5_image_dataset(upload_test+'/Test_%d.txt'%index, image_shape=(448, 448), mode='file', output_path='TEST_FIlE/upload_test_448_%d.h5'% index, categorical_labels=True, normalize=False)
    print 'Done creating new_train_448.h5'

