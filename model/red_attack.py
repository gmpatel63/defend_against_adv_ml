import numpy as np
import tensorflow as tf
import nibabel as nib
import pandas as pd
import time
import logging
import copy
from pathlib import Path

from . import config
from .datasets import load_normalized_mri
from .attacks import get_anatomical_features

# #max_diff function returns the max pixel value of the difference of two image
# def max_diff(img1,img2):
#     img = img1 - img2
#     return np.amax(img)
  
# #moves the source image to the boundary of the target image
# def boundary_estimation(source, target, dmin, pred_func):
#     # print("entering boundary_estimation")
#     Ii = ((source + target)/2.0)
#     k = pred_func(Ii)
#     delta = max_diff(source, Ii)
#     Ia2 = source
#     Ib2 = target
#     p = Ib2
#     #doing while the value of delta is greater than the dmin
#     loop_count = 0
#     while (delta > dmin):
#         if (pred_func(Ib2) != k):
#             Ia2 = Ii
#         else:
#             Ib2 = Ii
#         Ii = ((Ia2+Ib2)/2.0)
#         k = pred_func(Ii)
#         delta = max_diff(Ia2,Ii)
#         loop_count += 1
#         # print(f"prediction for Ii: {pred_func(Ii)}")
#     # print(f"iterations for boundary estimation: {loop_count}")
#     return Ii
  
# #go_out function moves the image again inside the target class if in between optimisation the image comes out
# def go_out(iout, alpha, target, pred_func):
#     i_diff = target - iout
#     inew = iout
# #moving the image in the direction of target image until it's class becomes same as that of target image
#     while (pred_func(inew)!=pred_func(target)):
#         inew = inew + 0.1*(i_diff)  
#     return inew


# #array_diff function calculates the square of the sum each given pixel value in 30x30 image
# def array_diff(d1):
#     sumd1 = 0.0
#     for l in range(1):
#         for i in range(0,156):
#             for j in range(0,220):
#                 for k in range(0,172):
#                     d1[k][j][i][l] = d1[k][j][i][l]*d1[k][j][i][l]
#                     sumd1 = sumd1 + d1[k][j][i][l]
#     return (sumd1)

# #gradient estimation function finds the sign in which the adversarial image should move so as to
# #reduce it's distance from the source image
# def gradient_estimation(source, target, adversarial, n, theta, pred_func):
#     # print("\n\n-------entering gradient estimation---------")
#     Ia = source
#     Ib = target
#     Ii = copy.deepcopy(adversarial)
#     max_val = np.max(source)

#     Io = np.zeros((5903040))
#     #generating n random random integers between 0 and 2700 to set those pixel value as 255 so as 
#     #to generate a new random image
#     X = np.random.randint(0,5903040, size=n)
#     for i in X:
#         # max_val = max value of mri 
#         Io[i] = max_val
#     Io = Io.reshape((172, 220, 156, 1))
#     #using newly generated random image to create an image near to the adversarial image
#     Ii2 = Ii + theta*Io

#     diff2 = Ii2 - Ia
#     diff1 = Ii - Ia
#     #finding the distance of the source image from the images the adversarial one and the new one generated from it
#     d2 = array_diff(diff2)
#     d1 = array_diff(diff1)
#     print("-------exiting gradient estimation---------\n\n")
#     #if the new one has large distance then move in it's opposite direction else move in the direction of adversarial image
#     if (d2 > d1):
#         return (-1, Ii2)
#     elif (d1 > d2):
#         return (1, Ii2)
#     else:
#         return (0,Ii2)
      
# #efficient update function finally moves the adversarial image in the direction given by gradient estimation function
# def efficient_update(source, target, adversarial, I2, g, j, pred_func):
#     print("\n\n-------entering efficient update---------")
#     Ia = source 
#     Ib = target
#     Ii = adversarial
#     Ii2 = I2
#     # delta is the vector with direction to move
#     delta = g*(Ii2 - Ii)
#     l = j
#     #adding delta to the given adversarial image with a jump size
#     Inew = Ii + l*delta
#     print(f"prediction for fresh Inew: {pred_func(Inew)}")
#     diff1 = Inew - Ia
#     diff2 = Ii - Ia
#     #calculating the distance after every movement from each image to source image so we know that we are moving right
#     d1 = array_diff(diff1)
#     d2 = array_diff(diff2)
#     ii = 0
#     it = 0
#     # we are moving the image till the Inew is optimzed even little
#     while(d1 > d2):
#         # reducing the count of jump after every move so that we don't go much far
#         l = (l/2.0)
#         Inew = Ii + l*delta
#         if(pred_func(Inew)!=pred_func(target)):
#             Inew = go_out(Inew,0.01,target)
#         it = it + 1
#         d1 = array_diff(Inew-Ia)        
#         #if after 100 iteration we didn't reach then may be we are going in wrong direction so we are breaking
#         if(it>100):
#             break
          
          
#     if (d1 > d2):
#         print(ii)
#         ii = ii + 1
#         Inew = Ii

#     return Inew
  
 

  
         
# def red_attack(args, params, paths, model):
#     logging.info(f'Starting red attack for {args.model}')
    
#     test_df = pd.read_csv(paths['test_data'])
    
#     iter_num = 1
#     n = 5
#     theta = 0.196
#     j = 1.0
#     dmin = 1.0

####################################################################################################################


class MRI(object):
    def __init__(self, mri, anat_features = None, pred = None, mri_type = None) -> None:
        self.mri = np.copy(mri)
        self.mri_type = mri_type
        self.anat_features = anat_features
        self.pred = pred
        
class RedAttack(object):
    def __init__(self, args, paths, model) -> None:
        self.iter_num = 1
        self.n = 100
        self.theta = 0.196
        self.jump_size = 1.0
        self.dmin = 1.0
        self.df = pd.read_csv(paths['test_data'])
        self.model = model
        self.model_name = args.model
        self.args = args
        self.targetMRI = self.get_target_dict()
        self.output_dir = Path(paths['csv_dir'])


    def get_target_dict(self):
        # find out mri with max age prediction - that will be our target        
        if(self.model_name in ['cnn', 'context_aware']):
            max_pred = 0
            target_anat_features = []
            target_mri = None
            anat_features = None
            for index, sample in self.df.iterrows():
                mri = load_normalized_mri(sample.mri_path)
                mri = np.expand_dims(mri, axis=0)
                if self.args.with_anat_features:
                    anat_features = get_anatomical_features(sample)
                    input_tensor = [(mri, anat_features)]
                else:
                    input_tensor = mri
                prediction = self.model(input_tensor).numpy()[0][0]
                if prediction >= max_pred:
                    max_pred = prediction
                    target_mri = mri[0]
                    target_anat_features = anat_features if tf.experimental.numpy.any(anat_features) else None
                    
            assert target_mri.any()
            targetMRI = MRI(target_mri, target_anat_features, max_pred, 'target')
            logging.info(f'found target with age: {targetMRI.pred}')
        return targetMRI 


    def predict_mri(self, mriObj):
        if(mriObj.pred):
            return mriObj.pred

        mri = mriObj.mri
        mri = np.expand_dims(mri, axis=0)
        if tf.experimental.numpy.any(mriObj.anat_features):
            input_tensor = [(mri, mriObj.anat_features)]
        else:
            input_tensor = mri
        prediction = self.model(input_tensor).numpy()[0][0]
        return prediction
   
    
    def attack(self):
        iters_results = []
        for index, sample in self.df.iterrows():
            logging.info('')
            logging.info('')
            logging.info(f'starting with index: {index}')
            # source_mri_dict = { "mri_type": "source" }
            source_mri = load_normalized_mri(sample.mri_path)
            source_mri = np.expand_dims(source_mri, axis=0)
            # source_mri_dict['mri'] = np.copy(source_mri[0])
            anat_features = None
            if self.args.with_anat_features:
                anat_features = get_anatomical_features(sample)
                # source_mri_dict['anat_features'] = anat_features
                input_tensor = [(source_mri, anat_features)]
            else:
                input_tensor = source_mri
            source_pred = self.model(input_tensor).numpy()[0][0]
            sourceMRI = MRI(source_mri[0], anat_features, source_pred, 'source') 
            
            start_time = time.time()
            be_iterations, diff_src, final_pred = self.iteration(sourceMRI)
            
            iters_results.append({'source age': source_pred, 'output age': final_pred, 'iterations': be_iterations, 'distance': diff_src})
            if index == 2:
                break
            
        print(f'iters_results: {iters_results}')
        df = pd.DataFrame(iters_results)
        df_name = f'{self.args.attack}_{self.args.model_name}'
        df_path = Path(self.output_dir, df_name)
        df.to_csv(df_path, index=False)
        
        
    #iteration function runs the whole process again and again until the adversarial image is sufficiently optimized
    def iteration(self, sourceMRI):
        # targett_dict = copy.deepcopy(self.target_dict)
        # sourcee = copy.deepcopy(source_dict)

        for i in range(self.iter_num):
            
            beMRI, iter_num, pred_after_be = self.boundary_estimation(sourceMRI)
            
            # go_out is not useful for us because it's a regression
            # if prediction goes higher than target prediction, it will go into infinite loop
            goMRI, pred_after_go = self.go_out(beMRI)
            
            grad_direction, geMRI = self.gradient_estimation(sourceMRI, goMRI)
            
            euMRI = self.efficient_update(sourceMRI, beMRI, geMRI, grad_direction)
            # # print(f"prediction for targett after e_u in iteration: {pred_func(targett)}")
            # fin = targett
            # if(pred_func(targett)!=pred_func(target)):
            #     fin = go_out(targett,0.01,target)

            # # if we have optimized the adversarial image then use the new optimized image again as adversarial image
            # #and rerun the whole process
            # if(array_diff(fin-sourcee)<array_diff(adversarial_image-sourcee)):
            #     targett = fin
            final_pred = self.predict_mri(euMRI)
            logging.info(f'final prediction: {final_pred}')
            diff_src = np.sum(np.square(euMRI.mri - sourceMRI.mri))
        return iter_num, diff_src, final_pred
    
    
    # moves the source image to the boundary of the target image
    def boundary_estimation(self, sourceMRI):
        logging.info('')
        logging.info('\t---boundary estimation---')
        dmin = self.dmin
        target_mri = np.copy(self.targetMRI.mri)
        source_mri = np.copy(sourceMRI.mri)
        ii_mri = ((source_mri + target_mri)/2.0)
        iiMRI = MRI(ii_mri, sourceMRI.anat_features)
        k = self.predict_mri(iiMRI)
        
        delta = np.amax(source_mri - ii_mri)
        Ia2 = copy.deepcopy(sourceMRI)
        Ib2 = copy.deepcopy(self.targetMRI)
        
        # doing while the value of delta is greater than the dmin
        iter_num = 0
        while (delta > dmin):
            if (self.predict_mri(Ib2) != k):
                Ia2 = MRI(iiMRI.mri, iiMRI.anat_features)
            else:
                Ib2 = MRI(iiMRI.mri, iiMRI.anat_features)
            iiMRI.mri = ((Ia2.mri+Ib2.mri)/2.0)
            k = self.predict_mri(iiMRI)
            delta = np.amax(Ia2.mri - iiMRI.mri)
            iter_num += 1
        
        pred_after_be = self.predict_mri(iiMRI)
        logging.info(f'\tprediction after boundary estimation: {pred_after_be}')
        logging.info(f'\titerations for boundary_estimation: {iter_num}')
        return iiMRI, iter_num, pred_after_be
    
    #go_out function moves the image again inside the target class if in between optimisation the image comes out
    def go_out(self, iiMRI):
        logging.info('')
        logging.info('\t---go out---')
        i_diff = self.targetMRI.mri - iiMRI.mri
        Inew = copy.deepcopy(iiMRI)
        target_pred = self.predict_mri(self.targetMRI)
        inew_pred = self.predict_mri(Inew)
        #moving the image in the direction of target image until it's class becomes same as that of target image
        while (inew_pred < target_pred):
            Inew.mri = Inew.mri + (0.1 * i_diff)
            inew_pred = self.predict_mri(Inew)
        print(f"\tprediction after go_out: {inew_pred}")
        return Inew, inew_pred
    
    
    #gradient estimation function finds the sign in which the adversarial image should move so as to
    #reduce it's distance from the source image
    def gradient_estimation(self, sourceMRI, advMRI):
        logging.info('')
        logging.info('\t---gradient estimation---')
        
        ii2MRI = copy.deepcopy(advMRI)
        max_val = np.max(sourceMRI.mri)

        Io = np.zeros((5903040))
        #generating n random random integers between 0 and 2700 to set those pixel value as 255 so as 
        #to generate a new random image
        X = np.random.randint(0, 5903040, size=self.n)
        for i in X:
            # max_val = max value of mri
            Io[i] = max_val
        Io = Io.reshape((172, 220, 156, 1))
        #using newly generated random image to create an image near to the adversarial image
        ii2MRI.mri = advMRI.mri + (self.theta * Io)

        diff2 = ii2MRI.mri - sourceMRI.mri
        diff1 = advMRI.mri - sourceMRI.mri
        #finding the distance of the source image from the images the adversarial one and the new one generated from it
        d2 = np.sum(np.square(diff2))
        d1 = np.sum(np.square(diff1))
        #if the new one has large distance then move in it's opposite direction else move in the direction of adversarial image
        if (d2 > d1):
            return -1, ii2MRI
        elif (d1 > d2):
            return 1, ii2MRI
        else:
            return 0, ii2MRI
        
    
    def efficient_update(self, sourceMRI, beMRI, geMRI, grad_direction):
        logging.info('')
        logging.info('\t---efficient update---')

        # delta is the vector with direction to move
        delta = grad_direction * (geMRI.mri - beMRI.mri)
        l = self.jump_size
        
        #adding delta to the given adversarial image with a jump size
        inew_mri = beMRI.mri + l * delta
        inewMRI = MRI(inew_mri, sourceMRI.anat_features)
        diff1 = inewMRI.mri - sourceMRI.mri
        diff2 = beMRI.mri - sourceMRI.mri
        
        #calculating the distance after every movement from each image to source image so we know that we are moving right
        d1 = np.sum(np.square(diff1))
        d2 = np.sum(np.square(diff2))
        target_pred = self.predict_mri(self.targetMRI)
        
        ii = 0
        it = 0
        
        # we are moving the image till the inew_mri is optimzed even little
        while(d1 > d2):
            # reducing the count of jump after every move so that we don't go much far
            l = (l/2.0)
            inewMRI.mri = beMRI.mri + l * delta
            if(self.predict_mri(inewMRI) < target_pred):
                inewMRI, _ = self.go_out(inewMRI)
            it = it + 1
            d1 = np.sum(np.square(inewMRI.mri - sourceMRI.mri))     
            #if after 100 iteration we didn't reach then may be we are going in wrong direction so we are breaking
            if(it > 100):
                break
            
        if (d1 > d2):
            print(ii)
            ii = ii + 1
            inewMRI = copy.deepcopy(beMRI)

        return inewMRI
    