# Author: Lior Frenkel
# edited by roi zini & aviad

# used packs import:
import argparse
import numpy as np
import random
import math
import pickle
import os
from pathlib import Path
import csv
from scipy.io import wavfile

# visualization packs:
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

# files import:
import nprirgen
from yaml import safe_load

# configs' args upload:
stream = open("/home/dsi/ziniroi/roi-aviad/config_try.yaml", 'r')
dictionary = safe_load(stream)
data_init = dictionary.get("Data")

## argsparse arguments:
parser = argparse.ArgumentParser()
parser.add_argument('--mission', type=str, default='separation', metavar='N', help='separation , enhancement , dereverberation or cmask')

# Random number of speaker def:
parser.add_argument('--not_rand_speak', action='store_true', default=data_init.get("fixed_num_of_speakers"), help='if specified, num of speakers is not random (default:True)') #'True'
parser.add_argument('--max_speakers', type=int, default=data_init.get("max_speakers"), metavar='N', help='The max number of speakers if random(default: 2)') #2

# Specific number of speaker def:
parser.add_argument('--n_speak', type=int, default=data_init.get("spekers_number"), metavar='NS', help='Number of speakers if not random (default: 2)')

# Data settings:
parser.add_argument('--mode', type=str, default=data_init.get("data_mode"), help='Mode, i.e., train, val or test')
parser.add_argument('--num_results', type=int, default=data_init.get("num_results").get(data_init.get("data_mode")), metavar='NR', help='Number of results')
parser.add_argument('--add_noise', action='store_true', default=data_init.get("noise"), help='if specified, adding random noise to speakers  (default:True)')
parser.add_argument('--doa',help='save also the doa', type=bool, default=data_init.get("store_doa"))

# mics array settings:
parser.add_argument('--n_mics', type=int, default=data_init.get("num_of_mics"), metavar='NM', help='Number of microphones (default: 4)')
parser.add_argument('--mic_dists', type=list, default=data_init.get("mic_array_dist"), help='array distance')

# Used paths:
parser.add_argument('--s_path', type=str, default=data_init.get("path").get("raw"), help='Speakers .wav files directory (default: /mnt/dsi_vol1/shared/sharon_db/wsj)')
parser.add_argument('--n_path', type=str, default=data_init.get("path").get("raw_noise"), help='Noise .wav files directory (default: /mnt/dsi_vol1/shared/sharon_db/wham_noise)')
parser.add_argument('--results_path', type=str, default=data_init.get("results_path").get("raw"), help='results directory')

# more settings:
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
parser.add_argument('--sir',help='different power between speakers', type=int, default=data_init.get("sir"))
parser.add_argument('--max_rt60', type=int, default=data_init.get("max_rt60"))
parser.add_argument('--snr', type=int, default=data_init.get("snr"))


args = parser.parse_args()

random.seed(args.seed)


assert len(args.mic_dists) == args.n_mics-1 , "wrong mic_dists or n_mics"

def diff_sir(sigs): # I ADDED

        for i in range(1,len(sigs)):
            gain = np.sqrt(10 ** (-args.sir / 10) * np.power(np.std(sigs[0]), 2) / np.power(np.std(sigs[i]), 2))
            sigs[i] = gain * sigs[i]

        return sigs



class Room:
    '''definition of every aspect of the room, relevant functions and so on...'''
    def __init__(self,args):
        super().__init__()

        min_w = 4
        max_w = 8
        min_l = 4
        max_l = 8
        min_h = 2.5
        max_h = 3
        min_rt60 = 0.13 # less than 0.13 create NaN
        max_rt60 = args.max_rt60


        self.room_width = np.random.uniform(min_w, max_w)
        self.room_length = np.random.uniform(min_l, max_l)
        self.room_height = np.random.uniform(min_h, max_h)
        self.rt60 = np.random.uniform(min_rt60, max_rt60)

        print('Room shape is {0:.2f}x{1:.2f}x{2:.2f}'.format(self.room_width, self.room_length, self.room_height))

    def get_dims(self):
        return round(self.room_width, 2), round(self.room_length, 2), round(self.room_height, 2), self.rt60


class Array:
    def __init__(self, width, length, array_area_w=0.5, array_area_l=0.5, res=5):
        super().__init__()

        min_array_z = 1
        max_array_z = 1.7

        self.array_x = np.random.uniform(0.5 * width - array_area_w, 0.5 * width + array_area_w)
        self.array_y = np.random.uniform(0.5 * length - array_area_l, 0.5 * length + array_area_l)
        self.array_z = round(np.random.uniform(min_array_z, max_array_z), 2)
        self.n_mic = args.n_mics
        self.mic_dists = args.mic_dists # [args.mic_dist] * (args.n_mics-1)

        theta_opt = np.arange(0, 180, res)
        theta_inx = np.random.randint(len(theta_opt))
        self.array_theta = theta_opt[theta_inx]

        print('Array center was located in ({0:.2f},{1:.2f},{2:.2f}) with theta = {3}'
              .format(self.array_x, self.array_y, self.array_z, self.array_theta))



    def get_array_loc(self):
        receivers = []
        if not self.mic_dists:  # One mic
            receivers.append([round(self.array_x, 2), round(self.array_y, 2), self.array_z])
        else:
            radius1 = sum(self.mic_dists) / 2
            if self.array_theta <= 90:
                mic_x1 = self.array_x - radius1 * math.cos(math.radians(self.array_theta))
                mic_y1 = self.array_y - radius1 * math.sin(math.radians(self.array_theta))
                receivers.append([round(mic_x1, 2), round(mic_y1, 2), self.array_z])
                for dist in self.mic_dists:
                    mic_x = mic_x1 + dist * math.cos(math.radians(self.array_theta))
                    mic_y = mic_y1 + dist * math.sin(math.radians(self.array_theta))
                    receivers.append([round(mic_x, 2), round(mic_y, 2), self.array_z])
                    mic_x1 = mic_x
                    mic_y1 = mic_y
            else:
                mic_x1 = self.array_x - radius1 * math.cos(math.radians(180 - self.array_theta))
                mic_y1 = self.array_y + radius1 * math.sin(math.radians(180 - self.array_theta))
                receivers.append([round(mic_x1, 2), round(mic_y1, 2), self.array_z])
                for dist in self.mic_dists:
                    mic_x = mic_x1 + dist * math.cos(math.radians(180 - self.array_theta))
                    mic_y = mic_y1 - dist * math.sin(math.radians(180 - self.array_theta))
                    receivers.append([round(mic_x, 2), round(mic_y, 2), self.array_z])
                    mic_x1 = mic_x
                    mic_y1 = mic_y

        return self.array_x, self.array_y, self.array_z, self.array_theta, receivers


class Speakers:
    def __init__(self, args, max_speakers, width, length, height, array_x, array_y, array_z, array_theta, rt60, limit=0.5):
        super().__init__()

        self.args = args
        if args.not_rand_speak:
            self.N = args.n_speak
        else:
            self.N = np.random.randint(1, max_speakers + 1)
        if args.add_noise:
            self.N += 1
        self.width = width
        self.length = length
        self.height = height
        self.rt60 = rt60
        self.array_x = array_x
        self.array_y = array_y
        self.array_z = array_z
        self.array_theta = array_theta
        self.n_mic = self.args.n_mics
        self.mic_dists =  args.mic_dists # [self.args.mic_dist] * (self.args.n_mics-1)

        self.limit = limit

        if args.add_noise:
            print('Number of speakers is: ', self.N - 1)
        else:
            print('Number of speakers is: ', self.N)

    def find_r_theta(self, theta_opt):
        theta_inx = np.random.randint(len(theta_opt))
        speaker_theta = theta_opt[theta_inx]
        if speaker_theta < 180 and speaker_theta != 0 and speaker_theta != 90:
            y_limit = self.length - self.limit - self.array_y
            x_max = y_limit / (math.tan(math.radians(speaker_theta)))
            if speaker_theta < 90:
                x_limit = self.width - self.limit - self.array_x
            elif speaker_theta > 90:
                x_limit = -(self.array_x - self.limit)
            y_max = math.tan(math.radians(speaker_theta)) * x_limit
            if speaker_theta < 90:
                max_speaker_x = min([x_max, x_limit])
            elif speaker_theta > 90:
                max_speaker_x = max([x_max, x_limit])
            max_speaker_y = min([y_max, y_limit])
            max_speaker_r = math.sqrt(pow(max_speaker_x, 2) + pow(max_speaker_y, 2))
        elif speaker_theta > 180 and speaker_theta != 270:
            y_limit = -(self.array_y - self.limit)
            x_max = y_limit / (math.atan(math.radians(speaker_theta - 180)))
            if speaker_theta < 270:
                x_limit = -(self.array_x - self.limit)
            elif speaker_theta > 270:
                x_limit = self.width - self.limit - self.array_x
            y_min = math.tan(math.radians(speaker_theta - 180)) * x_limit
            if speaker_theta < 270:
                max_speaker_x = max([x_max, x_limit])
            elif speaker_theta > 270:
                max_speaker_x = min([x_max, x_limit])
            max_speaker_y = max([y_min, y_limit])
            max_speaker_r = math.sqrt(pow(max_speaker_x, 2) + pow(max_speaker_y, 2))
        elif speaker_theta == 0:
            max_speaker_r = self.width - self.limit - self.array_x
        elif speaker_theta == 90:
            max_speaker_r = self.length - self.limit - self.array_y
        elif speaker_theta == 180:
            max_speaker_r = self.array_x - self.limit
        elif speaker_theta == 270:
            max_speaker_r = self.array_y - self.limit

        mic2speaker_min = 1
        speaker_r = np.random.uniform(mic2speaker_min, max_speaker_r)

        if 0 < self.array_theta <= 90:
            speaker_x = self.array_x + speaker_r * math.cos(math.radians(speaker_theta))
            speaker_y = self.array_y + speaker_r * math.sin(math.radians(speaker_theta))
        elif 90 < self.array_theta <= 180:
            speaker_x = self.array_x - speaker_r * math.cos(math.radians(180 - speaker_theta))
            speaker_y = self.array_y + speaker_r * math.sin(math.radians(180 - speaker_theta))
        elif 180 < self.array_theta <= 270:
            speaker_x = self.array_x - speaker_r * math.cos(math.radians(speaker_theta))
            speaker_y = self.array_y - speaker_r * math.sin(math.radians(speaker_theta))
        else:
            speaker_x = self.array_x + speaker_r * math.cos(math.radians(180 - speaker_theta))
            speaker_y = self.array_y - speaker_r * math.sin(math.radians(180 - speaker_theta))
        speaker_z = round(np.random.uniform(1.3, 1.9), 2)

        if self.array_theta <= 90:
            if speaker_theta >= self.array_theta:
                speaker_theta_array = speaker_theta - self.array_theta
            else:
                speaker_theta_array = 360 - (self.array_theta - speaker_theta)
        else:
            if speaker_theta >= self.array_theta + 180:
                speaker_theta_array = speaker_theta - self.array_theta - 180
            else:
                speaker_theta_array = speaker_theta - self.array_theta + 180

        return round(speaker_r, 2), speaker_theta, speaker_theta_array, \
               round(speaker_x, 2), round(speaker_y, 2), speaker_z

    def get_first_speaker(self, res=5):
        theta_opt = np.arange(0, 360, res)
        speaker_r, s_theta, s_theta_array, speaker_x, speaker_y, speaker_z = self.find_r_theta(theta_opt)
        print('First speaker angle from array is: ', s_theta_array)
        print('First speaker radius from array is: {0:.2f}'.format(speaker_r))

        return speaker_r, s_theta, s_theta_array, speaker_x, speaker_y, speaker_z

    def get_speaker(self, angles, i, islast, res=5, speaker_dist=20):
        theta_opt = np.arange(0, 360, res)
        for a in angles:
            del_inx = np.where(((theta_opt < a + speaker_dist) & (theta_opt > a - speaker_dist))
                               | (theta_opt > 360 + a - speaker_dist) | (theta_opt < a - 360 - speaker_dist))
            theta_opt = np.delete(theta_opt, del_inx)
        speaker_r, s_theta, s_theta_array, speaker_x, speaker_y, speaker_z = self.find_r_theta(theta_opt)
        if (not self.args.add_noise) or (self.args.add_noise and not islast):
            print('Speaker {0} angle from array is: {1}'.format(i, s_theta_array))
            print('Speaker {0} radius from array is: {1:.2f}'.format(i, speaker_r))

        return speaker_r, s_theta, s_theta_array, speaker_x, speaker_y, speaker_z

    def get_speakers(self, mics_loc, i):
        r1, theta1, theta1_array, x1, y1, z1 = self.get_first_speaker() # org - r1, theta1, theta1_array, x1, y1, z1 = self.get_first_speaker()
        speakers_loc = []
        thetas = []
        thetas_array = []
        dists = []
        speakers_loc.append([x1, y1, z1])
        thetas.append(theta1)
        thetas_array.append(theta1_array)
        dists.append(r1)
        is_last = False
        for s in range(1, self.N):
            if s == self.N - 1:
                is_last = True
            r, theta, theta_array, x, y, z = self.get_speaker(thetas, s + 1, is_last)
            thetas.append(theta)
            thetas_array.append(theta_array)
            dists.append(r)
            speakers_loc.append([x, y, z])
        if self.args.add_noise:
            print('Speakers angles from array are: ', thetas_array[:-1])
            print('Speakers distances from array are: ', dists[:-1])
        else:
            print('Speakers angles from array are: ', thetas_array)
            print('Speakers distances from array are: ', dists)
        h_list,h_list_1_order = self.get_speakers_h(speakers_loc, mics_loc)
        speakers_id, speakers_path = self.get_speakers_id()
        speakers_wav_files = self.choose_wav(speakers_path)
        speakers_conv, _ = self.conv_wav_h(speakers_wav_files, h_list, i, dists,h_list_1_order,thetas_array)
        
        return self.N, dists, thetas_array, speakers_loc, speakers_id, h_list, speakers_conv

    # gives us h according to speaker location(relative to mics location)
    def get_speaker_h(self, speaker_x, speaker_y, speaker_z, receivers):
        room_measures = [self.width, self.length, self.height]
        source_position = [speaker_x, speaker_y, speaker_z]
        fs = 16000
        n = 2048
        h, _, _ = nprirgen.np_generateRir(room_measures, source_position, receivers, reverbTime=self.rt60,
                                          fs=fs, orientation=[self.array_theta, .0], nSamples=n, nOrder=-1)
        h_1_order, _, _ = nprirgen.np_generateRir(room_measures, source_position, receivers, reverbTime=0,
                                          fs=fs, orientation=[self.array_theta, .0], nSamples=n, nOrder=-1)

        return h,h_1_order

    def get_speakers_h(self, speakers_loc, mics_loc):
        speakers_h = []
        speakers_h_1_order = []
        for inx, speaker in enumerate(speakers_loc):
            h,h_1_order = self.get_speaker_h(speaker[0], speaker[1], speaker[2], mics_loc)
            speakers_h.append(h)
            speakers_h_1_order.append(h_1_order)
            """
            print('h of speaker number {0}: '.format(inx + 1), h)
            """
        return speakers_h,speakers_h_1_order

    def get_speakers_id(self):
        if self.args.mode == 'train' or self.args.mode == 'val':
            PATH = self.args.s_path + '/Train'
            ids = os.listdir(PATH)
            # ids.remove('Train.zip')
        elif self.args.mode == 'test':
            PATH = self.args.s_path + '/Test'
            ids = os.listdir(PATH)
            # ids.remove('female')
            # ids.remove('male')
        else:
            raise Exception("Chosen mode is not valid!!")
        if self.args.add_noise:
            id_list = random.choices(ids, k=self.N - 1)
        else:
            id_list = random.choices(ids, k=self.N)
        id_path = [os.path.join(PATH, id) for id in id_list]
        if self.args.add_noise:
            noise = Noise(self.args)
            id_path.append(noise.choose_noise_path())

        return id_list, id_path

    def choose_wav(self, id_path):
        wav_lists = []
        for path in id_path:
            wav_lists.append([os.path.join(path, wav_file) for wav_file in os.listdir(path)])
        wav_lists1 = []
        if self.args.add_noise:
            for wav_list in wav_lists[:-1]:
                wav_lists1.append([path for path in wav_list if os.path.splitext(path)[1] == '.wav'])
            wav_lists1.append(wav_lists[-1])
        else:
            for wav_list in wav_lists:
                wav_lists1.append([path for path in wav_list if os.path.splitext(path)[1] == '.wav'])
        wav_files = [random.choice(wav_list) for wav_list in wav_lists1]

        return wav_files


    def conv_wav_h(self, wav_files, h_list, i, dists,h_list_1_order,thetas_array):  # i = scenario number
        speakers_conv = []
        speakers_delay_conv = []
        if self.args.add_noise:
            noise_file = wav_files[-1]
            noise_h = h_list[-1]
            wav_files = wav_files[:-1]
            h_list = h_list[:-1]
        k = 0
        for wave_file, h ,h_1_order in zip(wav_files, h_list,h_list_1_order):
            fs, wave = wavfile.read(wave_file)
            wave = np.copy(wave)

            
   
            speaker_conv = []
            speaker_delayed_conv = []
            if h.ndim ==1:
                wav_h_conv = np.convolve(wave, h)
                speaker_conv.append(np.expand_dims(wav_h_conv, axis=0))
            else:
                for s_mic_h in h:
                    wav_h_conv = np.convolve(wave, s_mic_h)
                    speaker_conv.append(np.expand_dims(wav_h_conv, axis=0))

            speaker_conv_np = np.concatenate(speaker_conv)
            speakers_conv.append(speaker_conv_np)


            if h_1_order.ndim==1:
                h_delay = h_1_order
            else:
                h_delay = h_1_order[0]
            wave_h_delay_conv = np.convolve(wave, h_delay)
            speaker_delayed_conv.append(np.expand_dims(wave_h_delay_conv, axis=0))
            speaker_delayed_conv_np = np.concatenate(speaker_delayed_conv)
            speakers_delay_conv.append(speaker_delayed_conv_np)
            
            k += 1
        if self.args.add_noise:
            n_fs, noise = wavfile.read(noise_file)
            # noise,n_fs=sf.read(noise_file)
            noise = np.copy(noise[:, 0])  # Choose one noise channel

            noise_conv = []
            if noise_h.ndim ==1:
                noise_h_conv = np.convolve(wave, noise_h)
                noise_conv.append(np.expand_dims(noise_h_conv, axis=0))
            else:
                for n_mic_h in noise_h:
                    noise_h_conv = np.convolve(noise, n_mic_h)
                    noise_conv.append(np.expand_dims(noise_h_conv, axis=0))
            noise_conv_np = np.concatenate(noise_conv)
            speakers_conv.append(noise_conv_np)

        len_list = [arr.shape[1] for arr in speakers_conv]
        min_len = min(len_list)
        speakers_conv_cut = [arr[:, :min_len] for arr in speakers_conv]

        if self.args.sir >0:
            speakers_conv_cut[:-1] = diff_sir(speakers_conv_cut[:-1])
        if self.args.add_noise:
            noise_conv_cut = speakers_conv_cut[-1]
            mixed_sig_np = sum(speakers_conv_cut[:-1])
            mixed_sig_np = Noise(self.args).get_mixed(mixed_sig_np, noise_conv_cut)
        else:
            mixed_sig_np = sum(speakers_conv_cut)

        ## Normalizing result to between -1 and 1
        max_conv_val = np.max(mixed_sig_np)
        min_conv_val = np.min(mixed_sig_np)
        mixed_sig_np = 2*(mixed_sig_np-min_conv_val) / (max_conv_val-min_conv_val) -1

        idx = 0
        speakers_delay = []
        speakers_delay_conv_cut = [arr[:, :min_len] for arr in speakers_delay_conv]
        while idx <= len(speakers_delay_conv_cut) - 1:
            speaker_sig = speakers_delay_conv_cut[idx]
            # Normalizing target to between -1 and 1
            max_conv_val = np.max(speaker_sig)
            min_conv_val = np.min(speaker_sig)
            speaker_sig = 2*(speaker_sig-min_conv_val) / (max_conv_val-min_conv_val) -1
            speakers_delay.append(speaker_sig)  
            idx+=1    
        
        # speakers_delay_np=np.asarray(speakers_delay)

        conv_dir = os.path.join(args.results_path, args.mode)
        conv_path = os.path.join(conv_dir, args.mode + '_scenario_{0}.p'.format(i + 1)) 

        # project - roi&aviad - 09/01/2022 - too many wav files were save #
        # # project - roi&aviad - 29-30/12/2021 #
        # ######################################################################
        # # saves mixed signals from diffrent mics into .wav file:
        # wave_path = os.path.join(conv_dir, 'scenario_{0}_mixed.wav'.format(i+1))
        # wavfile.write(wave_path, fs, mixed_sig_np.T)
        #
        # ## saves clean delayed signals into .wav file:
        # #speakers_delay_np=np.array(speakers_delay)
        # #wave_path = os.path.join(conv_dir, 'scenario_{0}_cln_targets.wav'.format(i+1))
        # #wavfile.write(wave_path, fs,speakers_delay_np[:,0,:].T)
        # ######################################################################
        
        # # project - roi&aviad - 02/01/2021 #
        # ######################################################################
        # # saves clean delayed signals into 2 .wav files:
        # speakers_delay_np=np.array(speakers_delay)
        # for k in range(speakers_delay_np.shape[0]):
        #     wave_path = os.path.join(conv_dir, 'scenario_{0}_cln_target'.format(i+1)+'{0}.wav'.format(k+1))
        #     wavfile.write(wave_path, fs,speakers_delay_np[k,0,:].T)


        if not os.path.exists(conv_dir):
            os.mkdir(conv_dir)

        if self.args.doa:
            # save the needed args to a pickle object:
            with open(conv_path, 'wb') as f:
                pickle.dump((mixed_sig_np, speakers_delay,h_list,thetas_array), f)


        else:
            with open(conv_path, 'wb') as f:
                pickle.dump((mixed_sig_np, speakers_delay,h_list), f) # ,noise_conv_cut

            #wave_path = os.path.join(conv_dir, 'scenario_{0}_mixed.wav'.format(i + 1))
            #wavfile.write(wave_path, fs, mixed_sig_np)
            #wave_path = os.path.join(conv_dir, 'scenario_{0}_cln.wav'.format(i + 1))
            #wavfile.write(wave_path, fs, speakers_delay[0])#############
        return mixed_sig_np, conv_path


class Noise:
    def __init__(self, args):
        super().__init__()

        # self.snr = np.random.uniform(5, 15)
        self.snr = args.snr # 20
        self.args = args

    def choose_noise_path(self):
        if self.args.mode == 'train':
            path = self.args.n_path + '/tr'
        elif self.args.mode == 'val':
            path = self.args.n_path + '/cv'
        elif self.args.mode == 'test':
            path = self.args.n_path + '/tt'
        else:
            raise Exception("Chosen mode is not valid!!")

        return path

    def get_mixed(self, mixed, noise):
        noise_gain = np.sqrt(10 ** (-self.snr / 10) * np.power(np.std(mixed), 2) / np.power(np.std(noise), 2))
        noise = noise_gain * noise
        mixed = mixed + noise

        return mixed

##########################################################################
#def plot_room(args, width, length, height, speakers_loc, mic_xy, scenario):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    #ax.azim=0
#    ax.elev=15
#
#    plt.title('Scenario {0}'.format(scenario))
#    ax.set_xlim(0, width)
#    ax.set_ylim(0, length)
#    ax.set_zlim(0, height)
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    ax.set_zlabel('Z')
#    speakers_loc_np = np.array(speakers_loc)
#    mic_xy_np = np.array(mic_xy)
#    speakers_x = speakers_loc_np[:, 0]
#    speakers_y = speakers_loc_np[:, 1]
#    speakers_z = speakers_loc_np[:, 2]
#    mics_x = mic_xy_np[:, 0]
#    mics_y = mic_xy_np[:, 1]
#    mics_z = mic_xy_np[:, 2]
#    ax.scatter(speakers_x, speakers_y, speakers_z, marker='o', s=20, c='b', depthshade=True)
#    ax.scatter(mics_x, mics_y, mics_z, marker='x', s=20, c='r', depthshade=True)
#    plt.savefig('{0}/train/room.png'.format(args.results_path, scenario + 1))


def get_scenario_data(args, scenario):
    room = Room(args)
    w, l, h, rt60 = room.get_dims()
    array = Array(w, l)
    array_x, array_y, array_z, array_theta, mics_xy = array.get_array_loc()
    max_speakers = args.max_speakers
    speakers = Speakers(args, max_speakers, w, l, h, array_x, array_y, array_z, array_theta, rt60)
    num_speakers, dists, thetas, speakers_xy, speakers_id, h_list, conv = speakers.get_speakers(mics_xy, scenario)

    # plot_room(args, w, l, h, speakers_xy, mics_xy, scenario)########################################33

    return w, l, h, rt60, array_theta, mics_xy, speakers_xy, dists, thetas, speakers_id, h_list, conv


def create_csv_results(args):
    with open('{0}_'.format(args.results_path) + args.mode + '.csv', 'w', newline='') as file:
        first_row = ['scenario', 'room_x', 'room_y', 'room_z', 'rt60']
        first_row.append('num_mics')
        dims = ['x', 'y', 'z']
        for mic in range(args.n_mics):
            for dim in dims:
                first_row.append('mic{0}_{1}'.format(mic + 1, dim))
        first_row.append('array_theta')
        first_row.append('num_speakers')
        for s in range(args.max_speakers):
            first_row.append('speaker{0}_id'.format(s + 1))
            first_row.append('speaker{0}_radius'.format(s + 1))
            first_row.append('speaker{0}_doa'.format(s + 1))  # I changed from theta to doa
        for s in range(args.max_speakers):
            for dim in dims:
                first_row.append('speaker{0}_{1}'.format(s + 1, dim))
        writer = csv.DictWriter(file, fieldnames=first_row)
        writer.writeheader()

        for result in range(args.num_results):
            w, l, h, rt60, array_theta, mics_xy, speakers_xy, dists, thetas, speakers_id, _, _ = get_scenario_data(args,
                                                                                                                   result)
            row_dict = {'scenario': result + 1, 'room_x': w, 'room_y': l, 'room_z': h, 'rt60': round(rt60, 2)}
            mics_xy_np = np.array(mics_xy)
            row_dict['num_mics'] = args.n_mics
            for mic in range(args.n_mics):
                for dim_inx, dim in enumerate(dims):
                    if mics_xy_np.ndim == 1:
                        row_dict['mic{0}_{1}'.format(mic + 1, dim)] = mics_xy_np[dim_inx]
                    else:
                        row_dict['mic{0}_{1}'.format(mic + 1, dim)] = mics_xy_np[mic][dim_inx]
            row_dict['array_theta'] = array_theta
            speakers_xy_np = np.array(speakers_xy)
            if args.add_noise:
                speakers_xy = speakers_xy[:-1]
            row_dict['num_speakers'] = len(speakers_xy)
            for s in range(len(speakers_xy)):
                row_dict['speaker{0}_id'.format(s + 1)] = speakers_id[s]
                row_dict['speaker{0}_radius'.format(s + 1)] = dists[s]
                row_dict['speaker{0}_doa'.format(s + 1)] = thetas[s]  # I changed from theta to doa
            for s in range(len(speakers_xy)):
                for dim_inx, dim in enumerate(dims):
                    if speakers_xy_np.ndim == 1:
                        row_dict['speaker{0}_{1}'.format(s + 1, dim)] = speakers_xy_np[dim_inx]
                    else:
                        row_dict['speaker{0}_{1}'.format(s + 1, dim)] = speakers_xy_np[s][dim_inx]
            writer.writerow(row_dict)
        file.close()


if __name__ == "__main__":
    Path(args.results_path).mkdir(exist_ok=True)

    # SAVE ARGS
    args_dict = vars(args)  # change the args object to a dictionary object

    # settings file - need to be more readable
    file_name = args.results_path + '/' + args.mode + '_settings.txt'
    with open(file_name, 'wt') as opt_file:
        for k, v in sorted(
                args_dict.items()):  # items: make a dictionary as list of tuples that each tuple is a couple of (k,v)
            opt_file.write('%s: %s\n' % (str(k), str(v)))

    create_csv_results(args)
    print('done create '+ args.mode +' raw data ;)')
    # build_features_func_.build_features_func()