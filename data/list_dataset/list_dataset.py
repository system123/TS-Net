'''
Created on Nov 17, 2017

@author: en
'''

import os, sys
import numpy as np
from scipy.misc.pilutil import imread  # @UnresolvedImport
import shutil
from glob import glob
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')
from data.abstract_dataset import Abstract_dataset

class ListDataset(Abstract_dataset):
    '''
    classdocs
    '''

    def __init__(self, nb_fake_images=3, split_ratio=[0.1, 0.7, 0.2], version = 512, category='sen12'):
        '''
        Constructor
        '''
        self.working_dir = os.getcwd()
        self.category = category
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        Abstract_dataset.__init__(self, nb_fake_images, split_ratio)
        self.version = version
        os.chdir(self.working_dir)

    def _load_file_list(self, path):
        img_list = []
        for line in open(path, 'r'):
            img_list.append(line.strip())
        return(img_list)

    def get_patch(self, im_path, name):

        co = imread('./data/%s/opt/%s'%(self.category, im_path), 1).astype('uint8')
        ir = imread('./data/%s/sar/%s'%(self.category, im_path), 1).astype('uint8')

        patch = []
        
        xy = np.zeros((self.nb_fake_images +1, 2), dtype='uint8')
        
        for i in range(0, co.shape[0], 16):
            for j in range(0, co.shape[1], 16):
                
                if i +64 > co.shape[0] or j + 64 > co.shape[1]: continue
                
                left, right = co[i: i + 64, j:j + 64], ir[i: i + 64, j:j + 64]
#                 print (left.shape, right.shape)
                left, right, left_affine, right_affine = self.transform_patch(left, right)
                xy[:, :] = [i+32, j+32]

                patch.append([left, right, [name] * (self.nb_fake_images +1), xy, left_affine, right_affine])
    
        left, right, name, xy, left_affine, right_affine = zip(*patch)
        assert len(left) == len(right) and len(left[0]) == 4
    
        return [np.vstack(left), np.vstack(right), np.hstack(name), np.vstack(xy), np.vstack(left_affine), np.vstack(right_affine)]

    def process_patches(self, im_names, train_test_set):
        patch = [self.get_patch(e, os.path.splitext(e)[0]) for e in im_names]
        
        left, right, name, xy, left_affine, right_affine = zip(*[e for e in patch if e[0] is not None])
        left, right = np.vstack(left), np.vstack(right)
        name, xy = np.hstack(name), np.vstack(xy)
        left_affine, right_affine = np.vstack(left_affine), np.vstack(right_affine)
        self.convert_np_tf_record(left, right, name, xy, left_affine, right_affine, train_test_set, self.category)
        return self.calcualte_mean_std(left, right)

    
    def split_train_test(self):
        
        self.working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        try: os.mkdir('./tfrecord/')
        except: pass
        
        try: os.mkdir('./tfrecord/%s'%self.category)
        except: pass

        img_list = glob('./data/%s/opt/*.png'%self.category)
        image_files = np.array([os.path.basename(x) for x in img_list])

        n_img_files = image_files.shape[0]

        print ('\n==========================')
        print ('total number of images in ', self.category, ':', n_img_files)
        print ('nb of images in train set:', n_img_files * self.split_ratio[1])
        print ('nb of images in test set:', n_img_files * self.split_ratio[2])
        print ('nb of images in validation set:', n_img_files * self.split_ratio[0])
        print ('\n==========================')

        indx = np.random.randint(0, n_img_files, 5 * n_img_files)
        train_indx = np.unique(indx)[:int(n_img_files * self.split_ratio[1])]
        test_indx = np.array([i for i in range(n_img_files) if i not in train_indx])
        validation_indx = test_indx[:int(n_img_files * self.split_ratio[0])]
        test_indx = test_indx[int(n_img_files * self.split_ratio[0]):]
        
        print ('nb of train, test and validation images:', train_indx.shape[0], test_indx.shape[0], validation_indx.shape[0])

        assert train_indx.shape[0] + test_indx.shape[0] + validation_indx.shape[0] == image_files.shape[0]

        for each in train_indx:
            assert each not in test_indx
            assert each not in validation_indx
            
        for each in validation_indx:
            assert each not in test_indx
        
        print ('testing image index', test_indx.shape[0])
        self.train_images = image_files[train_indx]
        self.test_images = image_files[test_indx]
        self.valid_images = image_files[validation_indx]
        
        mean_std_train = self.process_patches(self.train_images, 'train')
        mean_std_test = self.process_patches(self.test_images, 'test')
        mean_std_valid = self.process_patches(self.valid_images, 'validation')

        mean_std = (mean_std_train + mean_std_test + mean_std_valid)/3
        
        np.save('./tfrecord/%s/std_mean'%(self.category), mean_std)
        os.chdir(self.working_dir)

    def combine_std_mean(self):
        
        self.working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        shutil.copy('./tfrecord/%s/std_mean.npy'%self.category, './tfrecord/')
        os.chdir(self.working_dir)


if __name__ == '__main__':
    
    cuhk = ListDataset(category='sen12')
    cuhk.split_train_test()
    cuhk.combine_std_mean()