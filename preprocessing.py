import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

import numpy as np
import librosa
import matplotlib
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
matplotlib.use('Agg')

sr = 160

# For one channel, 9759 frames of data, meaning 61 seconds
# So every 9760 / 12 (segments) is duration for 5 seconds.

# want 5 seconds

# First batch subject 1-11
# First batch subject 2-21
# First batch subject 3-31
# First batch subject 4-41
# ...
#
sub_start = 1
sub_end = 109+1

experiment_num = [5,6,7,8,9,10]


channel_categories = {"Fp":['Fp1.', 'Fpz.', 'Fp2.'],
                      "Af":['Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.'],
                      "F":['F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..'],
                      "Ft":['Ft7.', 'Ft8.'],
                      "Fc":['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.'],
                      "T":['T7..', 'T8..', 'T9..', 'T10.'],
                      "C":['C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..'],
                      "Tp":['Tp7.', 'Tp8.'],
                      "Cp": ['Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.'],
                      "P":['P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..'],
                      "Po":['Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.'],
                      "O":[ 'O1..', 'Oz..', 'O2..'],
                      "I":['Iz..']
                     }


cat_avg = {"Fp": None,"Af": None,"F": None,"Ft": None,"Fc": None,"T": None,"C": None,"Tp": None,"Cp": None,"P": None,"Po": None,"O": None,"I": None}

cat_seq = ["Fc","C","Cp","Fp","Af","F","Ft","T","Tp","P","Po","O","I"]

for run in experiment_num:
    for i in range(sub_start, sub_end):
        import os
        
        
        # Change path here for output folder
        os.makedirs('E:\\DeepLearning_AI_CW\\Experiment_'+str(run)+'_new_image_output_'+str(sub_start)+'-'+str(sub_end)+'\\subject_' + str(i) + '\\')
        #     From 0 to 64
        # Define the parameters
        subject = i  # use data from subject 1
        runs = [run]  # use only hand and feet motor imagery runs
        grouped_raw = []

        # Get data and locate in to given path
        files = eegbci.load_data(subject, runs, "E:\\DeepLearning_AI_CW\\files")
        # Read raw data files where each file contains a run
        raws = [read_raw_edf(f, preload=True) for f in files]
        # Combine all loaded runs
        raw_obj = concatenate_raws(raws)
        raw_data = raw_obj.get_data()

        df = raw_obj.to_data_frame()

        for channel_category in channel_categories:
            sum = 0
            cat_len = len(channel_categories[channel_category])
            for channel in channel_categories[channel_category]:
                sum += (df[channel] / 1000000)
            print(channel_category)
            print("Sum: ", sum, "\n")
            avg = sum / cat_len
            print("Avg: ", avg, "\n")
            cat_avg[channel_category] = avg

        for cat in cat_seq:
            grouped_raw.append(list(cat_avg[cat]))
            print("Category:", cat, ": ", list(cat_avg[cat])[:5])

        grouped_raw = np.asarray(grouped_raw)

        for j in range(13):
            start = 0
            duration = (int(9760 / 12))
            # Change path here for output folder
            os.makedirs('E:\\DeepLearning_AI_CW\\Experiment_'+str(run)+'_new_image_output_'+str(sub_start)+'-'+str(sub_end)+'\\subject_' + str(i) + '\\channel_' + str(j) + "\\")
            for segment in range(12):
                print("start", start, "duration", duration)
                print(grouped_raw[j][start:duration])
                duration_data = grouped_raw[j][start:duration]

                powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(duration_data, Fs=sr)
                plt.xticks([])
                plt.yticks([])
                plt.axis('off')
                plt.show()
                # Change path here for output folder
                plt.savefig("E:\\DeepLearning_AI_CW\\Experiment_"+str(run)+"_new_image_output_"+str(sub_start)+"-"+str(sub_end)+"\\subject_" + str(i) + "\\channel_" + str(j) + "\\example_" + str(
                        segment) + ".png", bbox_inches='tight', pad_inches=0, dpi=96)




                duration += 813
                start += 813
