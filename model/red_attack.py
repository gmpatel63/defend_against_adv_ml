import random
import numpy as np
import tensorflow as tf
import nibabel as nib
import pandas as pd
import time
import logging
import copy
from pathlib import Path
from datetime import datetime
import requests
import base64
import json

from skimage.metrics import structural_similarity as ssim
from . import config
from .datasets import load_normalized_mri, get_dataframe
from .attacks import get_anatomical_features


class MRI(object):
    def __init__(self, mri, anat_features=None, pred=None, mri_type=None) -> None:
        self.mri = np.copy(mri)
        self.mri_type = mri_type
        self.anat_features = anat_features
        self.pred = pred


class RedAttack(object):
    def __init__(self, args, paths, model) -> None:
        self.iter_num = 1000
        self.n = 10000
        # in noromalized MRI, magnitude of each pixel will vary
        # between approx -4 to 17. so thete needs to be small compared to
        # original paper, where theta was 5 but range of input was 0 to 255
        self.theta = 0.196
        self.jump_size = 1.0
        self.dmin = 0.5
        # get_dataframe is used when executing without srgan web server
        # self.df = get_dataframe(paths['test_data'], args)
        self.df = pd.read_csv(paths['test_data'])
        self.model = model
        self.model_name = args.model
        self.args = args
        self.targetMRI = self.get_target_dict()
        self.output_dir = Path(paths['csv_dir'], 'red_attack_results', datetime.now().strftime("%Y-%m-%d"))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.op_file_name = f'{time_str}_{self.args.attack}_{self.args.model}'
        self.df_path = Path(self.output_dir, f'{self.op_file_name}.csv')
        self.op_mri_path = Path(self.output_dir, f'{self.op_file_name}.nii')
        logging.info(f'params: n={self.n}, dmin={self.dmin}, theta={self.theta}, jump_size={self.jump_size}')

    def get_target_dict(self):
        # find out mri with max age prediction - that will be our target
        if(self.model_name in ['cnn', 'context_aware', 'srgan_cnn']):
            max_pred = 0
            target_anat_features = []
            target_mri = None
            for index, sample in self.df.iterrows():
                mri = load_normalized_mri(sample.mri_path)

                # srgan test
                if(self.model_name == 'srgan_cnn'):
                    mri = self.get_srgan_output(mri)

                mri = np.expand_dims(mri, axis=0)
                anat_features = []
                if self.args.with_anat_features:
                    anat_features = get_anatomical_features(sample)
                    input_tensor = [(mri, anat_features)]
                else:
                    input_tensor = mri
                prediction = self.model(input_tensor).numpy()[0][0]
                if prediction >= max_pred:
                    max_pred = prediction
                    target_mri = mri[0]
                    target_anat_features = anat_features if tf.experimental.numpy.any(
                        anat_features) else None

            assert target_mri.any()
            targetMRI = MRI(target_mri, target_anat_features,
                            max_pred, 'target')
            logging.info(f'found target with age: {targetMRI.pred}')
        return targetMRI

    def get_srgan_output(self, mri):
        # send request to srgan server
        srgan_url = "http://0.0.0.0:5050/"
        data = {"base64_mri": base64.b64encode(mri.tobytes()).decode('utf-8')}
        res = requests.post(srgan_url, json=data)
        res = res.json()
        # print(f"len of response mri string: {len(res['base64_mri'])}")
        mri_fl32 = np.frombuffer(base64.b64decode(
            res["base64_mri"]), dtype='float32').reshape((172, 220, 156))
        # srgan_mri = np.fromstring(res["base64_mri"],dtype='float64').reshape((172, 220, 156))
        srgan_mri = np.float64(mri_fl32)
        srgan_mri = np.expand_dims(srgan_mri, axis=3)
        return srgan_mri

    def predict_mri(self, mriObj):
        if(mriObj.pred):
            return mriObj.pred

        mri = mriObj.mri

        if(self.model_name == 'srgan_cnn'):
            mri = self.get_srgan_output(mri)

        mri = np.expand_dims(mri, axis=0)
        if mriObj.anat_features is not None and tf.experimental.numpy.any(mriObj.anat_features):
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

            if(self.model_name == 'srgan_cnn'):
                source_mri = self.get_srgan_output(source_mri)
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
            sourceMRI = MRI(
                source_mri[0], anat_features, source_pred, 'source')

            start_time = time.time()
            attack_result = self.iteration(sourceMRI, src_index=index)
            logging.info(
                f'time to attack one image: {time.time() - start_time}')

            # add a row for each iteration of boundary estimation
            iters_results.extend(attack_result)
            df = pd.DataFrame(iters_results)
            df.to_csv(self.df_path, index=False)
            
    
    def mri_ssim(self, source_mri, adv_mri):
        return ssim(source_mri, adv_mri, channel_axis=-1, gaussian_weights=True,
                       sigma=1.5, use_sample_covariance=False, data_range=np.max(source_mri) - np.min(source_mri))


    # iteration function runs the whole process again and again until the adversarial image is sufficiently optimized
    def iteration(self, sourceMRI, src_index):
        # targett_dict = copy.deepcopy(self.target_dict)
        # sourcee = copy.deepcopy(source_dict)
        beInputMRI = MRI(self.targetMRI.mri, self.targetMRI.anat_features)
        
        beMRI, be_results, pred_after_be = self.boundary_estimation(
                sourceMRI)
        be_l2_norm = np.sqrt(np.sum(np.square(beMRI.mri - sourceMRI.mri)))
        logging.info(f'be mri l2 norm: {be_l2_norm}')
        
        minNormMRI = copy.deepcopy(beMRI)
        
        iter_results = []
        for i in range(self.iter_num):
            logging.info(f'iteration: {i}')

            # go_out is not useful for us because it's a regression
            # if prediction goes higher than target prediction, it will go into infinite loop
            # goMRI, pred_after_go = self.go_out(beMRI)

            grad_direction, geMRI = self.gradient_estimation(sourceMRI, beMRI)
            ge_l2_norm = np.sqrt(np.sum(np.square(geMRI.mri - sourceMRI.mri)))
            logging.info(f'ge mri l2 norm: {ge_l2_norm}')

            euMRI, eu_iterations = self.efficient_update(
                sourceMRI, beMRI, geMRI, grad_direction)

            finMRI = copy.deepcopy(euMRI)
            eu_l2_norm = np.sqrt(np.sum(np.square(euMRI.mri - sourceMRI.mri)))
            logging.info(f'eu mri l2 norm: {eu_l2_norm}')
            
            logging.info(f'minNormMRI pred: {self.predict_mri(minNormMRI)}')
            logging.info(f'minNormMRI l2 norm: {np.sum(np.square((minNormMRI.mri - sourceMRI.mri)))}')
            
            # if we have optimized the adversarial image then use the new optimized image again as adversarial image
            # and rerun the whole process
            fin_l2_norm = np.sum(np.square((finMRI.mri - sourceMRI.mri)))
            be_l2_norm = np.sum(np.square((beMRI.mri - sourceMRI.mri)))
            # be_ssim = self.mri_ssim(sourceMRI.mri, beMRI.mri)
            # fin_ssim = self.mri_ssim(sourceMRI.mri, finMRI.mri)
            fin_pred = int(self.predict_mri(finMRI))
            target_pred = int(self.predict_mri(self.targetMRI))
            
            if(fin_l2_norm < be_l2_norm):
                
                logging.info("---- l2 norm decreased, using new MRI as finMRI ------")
                logging.info(f"prediction of l2 decreased MRI: {self.predict_mri(finMRI)}") 
                
                if(fin_pred<target_pred):
                    logging.info(f"prediction of fin mri is less than target, calling boundary estimation") 
                    finMRI, *_ = self.boundary_estimation(sourceMRI)
                    
                    fin_l2_norm = np.sum(np.square((finMRI.mri - sourceMRI.mri)))
                    fin_pred = int(self.predict_mri(finMRI))
                    # fin_ssim = self.mri_ssim(sourceMRI.mri, finMRI.mri)
                    if (fin_l2_norm < be_l2_norm and fin_pred >=target_pred):
                        logging.info(f"l2 norm after BE is smaller, updating minNormMRI") 
                        minNormMRI = copy.deepcopy(finMRI)
                    else:
                        logging.info(f"discarding this iterationm using minNormMRI") 
                        finMRI = copy.deepcopy(minNormMRI)
                else:
                    minNormMRI = copy.deepcopy(finMRI)
                    
                beMRI = finMRI
                
            
            iter_pred = self.predict_mri(beMRI)
            sm = self.mri_ssim(sourceMRI.mri, beMRI.mri)
            # logging.info(f'sm: {sm}')
            l2_norm = np.sqrt(np.sum(np.square(beMRI.mri - sourceMRI.mri)))
            # logging.info(
            #     f'iteration: {i}, iter pred: {iter_pred}, sm: {sm}, l2_norm: {l2_norm}')
            logging.info("")
            logging.info("")
            iter_results.append({'idx': src_index, 'iteration': i, 'output_pred': iter_pred, 'ssim': sm, 'l2_norm': l2_norm})
            
        if src_index == 0:
            op_mri = np.squeeze(beMRI.mri)
            op_mri = nib.Nifti1Image(op_mri, np.eye(4))
            op_mri.to_filename(self.op_mri_path)

        return iter_results

    # moves the source image to the boundary of the target image

    def boundary_estimation(self, sourceMRI, targetMRI=None):
        # logging.info('')
        # logging.info('\t---boundary estimation---')
        
        dmin = self.dmin
        
        if not targetMRI or int(self.predict_mri(targetMRI)) < int(self.predict_mri(targetMRI)):
            targetMRI = MRI(self.targetMRI.mri, self.targetMRI.anat_features)
        
        target_mri = np.copy(targetMRI.mri)
        source_mri = np.copy(sourceMRI.mri)

        ii_mri = ((source_mri + target_mri)/2.0)
        iiMRI = MRI(ii_mri, sourceMRI.anat_features)
        
        k = self.predict_mri(iiMRI)

        delta = np.amax(source_mri - iiMRI.mri)
        Ia2 = copy.deepcopy(sourceMRI)
        Ib2 = copy.deepcopy(targetMRI)

        # doing while the value of delta is greater than the dmin
        iter_num = 0
        be_results = []
        while (delta > dmin):
            if (int(self.predict_mri(Ib2)) != int(k)):
                Ia2 = MRI(iiMRI.mri, iiMRI.anat_features)
            else:
                Ib2 = MRI(iiMRI.mri, iiMRI.anat_features)
            iiMRI.mri = ((Ia2.mri+Ib2.mri)/2.0)
            k = self.predict_mri(iiMRI)
            delta = np.amax(Ia2.mri - iiMRI.mri)
            iter_num += 1
            be_results.append({'iteration': iter_num, 'prediction': k})

        pred_after_be = self.predict_mri(iiMRI)
        # logging.info(
        #     f'\tprediction after boundary estimation: {pred_after_be}')
        # logging.info(f'\titerations for boundary_estimation: {iter_num}')
        return iiMRI, be_results, pred_after_be

    # go_out function moves image just out of class boundary
    def go_out(self, iiMRI):
        logging.info('\t---go out---')
        i_diff = self.targetMRI.mri - iiMRI.mri
        Inew = copy.deepcopy(iiMRI)
        target_pred = self.predict_mri(self.targetMRI)
        inew_pred = self.predict_mri(Inew)
        # moving the image in the direction of target image until it's class becomes same as that of target image
        while (inew_pred < target_pred):
            # print(f'in go_out')
            Inew.mri = Inew.mri + (0.1 * i_diff)
            inew_pred = self.predict_mri(Inew)
        logging.info(f"\tprediction after go_out: {inew_pred}")
        return Inew, inew_pred

    # gradient estimation function finds the sign in which the adversarial image should move so as to
    # reduce it's distance from the source image

    def gradient_estimation(self, sourceMRI, advMRI):
        # logging.info('')
        # logging.info('\t---gradient estimation---')

        ii2MRI = copy.deepcopy(advMRI)
        max_val = np.max(sourceMRI.mri)
        min_val = np.min(sourceMRI.mri)

        Io = np.zeros((5903040))
        # generating n random random integers between 0 and 2700 to set those pixel value as 255 so as
        # to generate a new random image
        X = np.random.randint(0, 5903040, size=self.n)
        for i in X:
            # max_val = max value of mri
            # Io[i] = random.uniform(min_val, max_val)
            Io[i] = max_val
        Io = Io.reshape((172, 220, 156, 1))
        # using newly generated random image to create an image near to the adversarial image
        ii2MRI.mri = advMRI.mri + (self.theta * Io)
        
        ii2MRINew = ii2MRI
        # if(int(self.predict_mri(ii2MRINew))<int(self.predict_mri(self.targetMRI))):
        #    ii2MRINew, *_ = self.boundary_estimation(sourceMRI, ii2MRI)
            # ii2MRINew, *_ = self.go_out(ii2MRINew)

        diff2 = ii2MRINew.mri - sourceMRI.mri
        diff1 = advMRI.mri - sourceMRI.mri
        # finding the distance of the source image from the images the adversarial one and the new one generated from it
        d2 = np.sum(np.square(diff2))
        d1 = np.sum(np.square(diff1))
        # if the new one has large distance then move in it's opposite direction else move in the direction of adversarial image
        if (d2 > d1):
            return -1, ii2MRINew
        elif (d1 > d2):
            return 1, ii2MRINew
        else:
            return 0, ii2MRINew

    def efficient_update(self, sourceMRI, beMRI, geMRI, grad_direction):
        # logging.info('\t---efficient update---')

        # delta is the vector with direction to move
        delta = grad_direction * (geMRI.mri - beMRI.mri)
        l = self.jump_size

        # adding delta to the given adversarial image with a jump size
        inew_mri = beMRI.mri + l * delta
        inewMRI = MRI(inew_mri, sourceMRI.anat_features)
        diff1 = inewMRI.mri - sourceMRI.mri
        diff2 = beMRI.mri - sourceMRI.mri

        # calculating the distance after every movement from each image to source image so we know that we are moving right
        d1 = np.sum(np.square(diff1))
        d2 = np.sum(np.square(diff2))
        target_pred = self.predict_mri(self.targetMRI)
        # logging.info(f'd1: {d1}, d2: {d2}')

        ii = 0
        iter_count = 0

        # we are moving the image till the inew_mri is optimzed even little
        while(d1 > d2):
            # reducing the count of jump after every move so that we don't go much far
            # l = (l/2.0)
            inewMRI.mri = beMRI.mri + l * delta
            # if(self.predict_mri(inewMRI) < target_pred):
            #     inewMRI, *_ = self.go_out(inewMRI)
            iter_count = iter_count + 1
            d1 = np.sum(np.square(inewMRI.mri - sourceMRI.mri))
            # if(iter_count % 100 == 0):
            #     current_ssim = ssim(sourceMRI.mri, inewMRI.mri, channel_axis=-1, gaussian_weights=True,
            #            sigma=1.5, use_sample_covariance=False, data_range=np.max(sourceMRI.mri) - np.min(sourceMRI.mri))
            #     logging.info(
            #         f'eu: iter_count: {iter_count}, ssim: {current_ssim}, d1: {d1}, d2: {d2}')
            if(iter_count > 20):
                break

        # if (d1 > d2):
        #     logging.info(f'eu didnt work, sending back beMRI')
        #     ii = ii + 1
        #     iter_count = -1
        #     inewMRI = copy.deepcopy(beMRI)

        return inewMRI, iter_count
