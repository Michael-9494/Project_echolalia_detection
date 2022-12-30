# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:56:39 2022

@author: 97254
"""
import numpy as np
from matplotlib import patches
from numba import jit
import scipy
import matplotlib.pyplot as plt
import glob
import pandas as pd
import parselmouth 
import statistics

from matplotlib.colors import LinearSegmentedColormap

from parselmouth.praat import call
from scipy.stats.mstats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def enframe(x, overlap, WindowLength, Fs):
        
        
    # d_frames_r = enframe(cluster_frameR, 50, 1, fps)
    # one_d_frames_r = d_frames_r.reshape(1,-1)
    # ZeroCrossingSignal_R_clust = calcZCR(d_frames_r)
    # one_d_frames_b = enframe(frames_b[:, 100, 100], 50, 1, fps)
    # ZeroCrossingSignal_B = calcZCR(one_d_frames_b)
    # one_d_frames_g = enframe(frames_g[:, 100, 100], 50, 1, fps)
    # ZeroCrossingSignal_G = calcZCR(one_d_frames_g)
    
    # split signal up into (overlapping) frames: one per row.
    nx = len(x[:]);  # the signal length
    N = WindowLength * Fs  # [sec]*[sample/sec]=[sample]. samples in each row
    inc = ((overlap) * N) // 100
    nf = int((nx - N + inc) / inc);  # the number of segmants    
    
    f = np.zeros((int(nf), int(N)), 'int32')  # the segmants matrix
    indf = ((inc * (np.arange(nf))).T).astype(int)  # the initial index of the segment
    inds = (np.arange(N)).astype(int);  # numbers between 1-the segment length
    e_row = np.repeat(indf[:,None],int(N),axis = 1) 
    e_col = np.repeat(inds[None,:],int(nf),axis = 0)
    ind_mat = e_row + e_col
#
    f = x[ind_mat] *np.hamming(N)
         
    return f


def first_PreProcess(Signal,Fs,alpha,WindowLength,Overlap):
    
    """
     first_PreProcess make all the pre processing wotk we need to do
     the steps are based on DC removal, HP filtering , framing and multiply
     each segment with hanning filter
     
    Args:
        Signal - the raw speech signal; Fs- Sampling frequency
        alpha - pre-emphasis filter parameter;
        WindowLength - window length [seconds];
        Overlap - percentage of overlap between adjacent frames [0-100]
        


    Returns:
        ProcessedSig -the preprocessed speech signal
        FramedSig -a matrix of the framed signal (each row is a frame)
        
    """
    N = WindowLength*Fs; # [sec]*[sample/sec]=[sample]
    # Remove DC noise:
    Signal = Signal - np.mean(Signal);
    Signal = Signal/max(abs(Signal));
    # Pre-Emphasis filtering  
    ProcessedSig = librosa.effects.preemphasis(Signal, coef=alpha)
    FramedSig = enframe(ProcessedSig ,((Overlap)*N)//100, WindowLength, Fs  )
    return ProcessedSig,FramedSig


    
def stft(Signal, window, Hopsize, only_positive_frequencies=False):
    """
    Compute a  short-time Fourier transform (STFT)

    Args:
        Signal (np.ndarray)- Signal to be transformed
        window (np.ndarray)- window function
        Hopsize (int) - Hopsize 
        only_positive_frequencies (bool)- Return only positive frequency part of spectrum (non-invertible)
            (Default value = False)

    Returns:
        X (np.ndarray): The discrete short-time Fourier transform
    """
    N = len(window)
    L = len(Signal)
    M = int((L - N + Hopsize) / Hopsize)  # the number of segmants 
    X = np.zeros((N, M), dtype='complex')
    for m in range(M):
        x_windowin = Signal[m * Hopsize:m * Hopsize + N] * window
        X_windowin = np.fft.fft(x_windowin)
        X[:, m] = X_windowin

    if only_positive_frequencies:
        K = 1 + N // 2
        X = X[0:K, :]
    return X



def rmse(signal, frame_size, hop_length):
    rmse = []
    
    # calculate rmse for each frame
    for i in range(0, len(signal), hop_length): 
        rmse_current_frame = np.sqrt(sum(signal[i:i+frame_size]**2) / frame_size)
        rmse.append(rmse_current_frame)
    return np.array(rmse) 

    




# This is the function to measure source acoustics using default male parameters.

def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return duration,meanF0,stdevF0,hnr,localJitter,localabsoluteJitter,rapJitter,ppq5Jitter,ddpJitter,localShimmer,localdbShimmer,apq3Shimmer,aqpq5Shimmer,apq11Shimmer,ddaShimmer

# This function runs a 2-factor Principle Components Analysis (PCA) on Jitter and Shimmer
def runPCA(df):
    # z-score the Jitter and Shimmer measurements
    measures = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
                'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    x = df.loc[:, measures].values
    x = StandardScaler().fit_transform(x)
    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['JitterPCA', 'ShimmerPCA'])
    principalDf
    return principalDf


def draw_spectrogram(spectrogram, dynamic_range=50): 
    X, Y = spectrogram.x_grid(),spectrogram.y_grid() 
    sg_db = 10 * np.log10(spectrogram.values) 
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max()- dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax]) 
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")
    
def draw_intensity(intensity): 
        plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w') 
        plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
        plt.grid(False) 
        plt.ylim(0)
        plt.ylabel("intensity [dB]")
        
def draw_pitch(pitch): 
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot 
    pitch_values = pitch.selected_array['frequency'] 
    pitch_values[pitch_values==0] = np.nan 
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w', label=" pitch")
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2) 
    plt.grid(False) 
    plt.ylim(0, pitch.ceiling) 
    plt.ylabel("fundamental frequency [Hz]")
    
def quantize_matrix(C, quant_fct=None):
    """Quantize matrix values in a logarithmic manner (as done for CENS features)
    Args:
        C (np.ndarray): Input matrix
        quant_fct (list): List specifying the quantization function (Default value = None)

    Returns:
        C_quant (np.ndarray): Output matrix
    """
    C_quant = np.empty_like(C)
    if quant_fct is None:
        quant_fct = [(0.0, 0.05, 0), (0.05, 0.1, 1), (0.1, 0.2, 2), (0.2, 0.4, 3), (0.4, 1, 4)]
    for min_val, max_val, target_val in quant_fct:
        mask = np.logical_and(min_val <= C, C < max_val)
        C_quant[mask] = target_val
    return C_quant
    
@jit( forceobj=True)    
def process_frames(ADOS_speaker,sound,window_length,time_step):
    
    out_dictionery = {'F1':{},'F2':{},'F3':{},
                      'intensity':{},
                      'time_f':{},'Event':{},'Speaker':{},
                      'speech_data_time':{},'speech_data':{},
                      'pitch':{},'spectrogram':{},'pitch_obj':{},
                      'sound_obj':{},'f':{},'t':{},'t_spect':{},'warped_speech_data':{}}
    # 
    for i in ADOS_speaker.index:
            
            snd_part = sound.extract_part(from_time=ADOS_speaker.loc[i,0],
                                                         to_time =ADOS_speaker.loc[i,1], preserve_times=True )
        
            # snd_part = sound.extract_part(from_time=d['Start_time'][0],to_time =d['End_time'][0], preserve_times=True )
            out_dictionery['speech_data_time'][i]=snd_part.xs()
            out_dictionery['speech_data'][i]=snd_part.values.T  
            out_dictionery['sound_obj'][i]=snd_part
            
            # ch = out_dictionery['speech_data'][i]
            # fft_speech = np.fft.fft(ch)/len(ch)   #distribute energy
            # fft_speech_Single_side = fft_speech[0:np.floor(len(ch)/2).astype(int)].copy() # get one-side spectrum  
            # # fft_speech_Single_side2 = fft_speech_Single_side[1:-1] 
            # fft_speech_Single_side[1:-1] = 2*fft_speech_Single_side[1:-1]    # % Multiple by 2 as a correction for amplitude
            # warpedFrame = VTLN(fft_speech_Single_side, alpha=1.12)        
            # warpedFrame[1:-1] = fft_speech_Single_side[1:-1]/2              # % Devide by 2 as a correction for amplitude
            # warpedSpeech =np.fft.ifft(np.concatenate( (warpedFrame , warpedFrame[::-1,:].conj()))*len(ch))
            # warpedSpeech = np.real(warpedSpeech)
            # out_dictionery['warped_speech_data'][i] = warpedSpeech
            
            
            
            # 0:    time_step(s) (standard value: 0.0)
            # the measurement interval (frame duration), in seconds.
            # If you supply 0, Praat will use a time step of 0.75 / (pitch fide),
            # e.g. 0.01 seconds if the pitch floor is 75 Ha, in this example,
            # Praat computes 100 pitch values per second.
            
            # f0min pitch_floor(Hz) (standard value: 75 Hz)
            # candidates below this frequency will not be recuted.
            # This parameter determines the effective length of the analysis window: it will
            # be 3 longest periods long, ie., if the pitch floor is 75 Hz, 
            # the window will be effectively 3/75 = 0.04 seconds long.
            # Note that if you set the time step to zero, the analysis windows
            # for consecutive measurements will overlap appreciably: Praat will always
            # compute 4 pitch values within one window length, i.e., the degree of oversampling is 4.
            
            
            # 15 
            
            # 'no'       very_accurate
            
            #  0.03  silence_threshold  (standard value: 0.03)
            # frames that do not contain amplitudes above this threshold 
            # (relative to the global maximum amplitude), are probably silent.    
            
            
            #  0.45  Voicing_th (standard value: 0.45)
            # the strength of the unvoiced candidate,
            # relative to the maximum possible autocorrelation.
            # If the amount of periodic energy in a frame is more than this 
            # of the total energy (the remainder being noise), then Praat will prefer
            # to regard this frame as voiced; otherwise as unvoiced. To increase the
            # number of unvoiced decisions, increase the voicing threshold.    
            
            #  0.01   Octave_cost(standard value: 0.01 per octave)
            # degree of favouring of high-frequency candidates, relative to the maximum possible autocorrelation.
            # This is necessary because even (or: especially) in the case of a 
            # perfectly periodic signal, all undertones of Fo are equally strong
            # candidates as Fo itself. To more strongly favour recruitment of high-frequency 
            # candidates, increase this value
         
            #  0.35   octave-Jump cost    (standard value: 0.35)
            # degree of disfavouring of pitch changes, relative to the maximum 
            # possible autocorrelation. To decrease the number of large frequency jumps,
            # increase this value. In contrast with what is described in the article,
            # this value will be corrected for the time step:
            # multiply by 0.01 s / TimeStep to get the value in the way it is 
            # used in the formulas in the article.       
            
            
            #  0.14  Voiced/Un-Voiced cost (standard value: 0.14)
            # degree of disfavouring of voiced/unvoiced transitions, relative to 
            # the maximum possible autocorrelation. To decrease the number of
            # voiced/unvoiced transitions, increase this value. In contrast with what
            # is described in the article, this value will be corrected for the time
            # step: multiply by 0.01s / TimeStep to get the value in the way it is
            # used in the formulas in the article.
            
            #  f0max
            pitch = call(snd_part, "To Pitch (cc)", 0,              # time_step(s)
                         60,                                        # f0min pitch_floor(Hz)
                         15,
                         'no',
                         0.03,                                      # silence_threshold
                         0.45,                                      # Voicing_th               
                         0.01,
                         0.35,
                         0.14,
                         1600)                                      #  f0max
            # pitch = snd_part.to_pitch()
            pitch_values = pitch.selected_array['frequency'] 
            pitch_values[pitch_values==0] = np.nan
            out_dictionery['pitch'][i] = pitch_values
            out_dictionery['pitch_obj'][i] = pitch
            # pre_emphasized_sound = snd_part.copy() 
            # pre_emphasized_sound.pre_emphasize() 
            spectrogram = snd_part.to_spectrogram(window_length=window_length,maximum_frequency=8000)
            t, f = spectrogram.x_grid(),spectrogram.y_grid()
            out_dictionery['f'][i] = f
            out_dictionery['t_spect'][i] = t
            # sg_db = 10 * np.log10(spectrogram.values)
            out_dictionery['spectrogram'][i] = spectrogram
            out_dictionery['Event'][i]=ADOS_speaker[3][i]
            out_dictionery['Speaker'][i]=ADOS_speaker[2][i]
            intensity = snd_part.to_intensity()
            out_dictionery['intensity'][i]=intensity
            f0min=60
            f0max=1600
            pointProcess = call(snd_part, "To PointProcess (periodic, cc)",
                                f0min,
                                f0max)
            numPoints = call(pointProcess, "Get number of points")
               
            formants = call(snd_part, "To Formant (burg)",
                            time_step,                                  # time step(s),
                            5,  
                            8000,                                       # formant ceiling(Hz),
                            window_length,                              # window length(s),
                            50)                                         # Pre-emphasis from(Hz)
            
            f1_list = []
            f2_list = []
            f3_list = []
            f4_list = []
            time_f = []
            
                # Measure formants only at glottal pulses
            for point in range(0, numPoints):
                point += 1
                t = call(pointProcess, "Get time from index", point)
                f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
                f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
                f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
                f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
                time_f.append(t)
                f1_list.append(f1)
                f2_list.append(f2)
                f3_list.append(f3)
                f4_list.append(f4)
                    
            if f1_list:
                f1_list_no_nan = [f1 for f1 in f1_list if str(f1) != 'nan']
                f2_list_no_nan = [f2 for f2 in f2_list if str(f2) != 'nan']
                f3_list_no_nan = [f3 for f3 in f3_list if str(f3) != 'nan']
               
                out_dictionery['time_f'][i]=time_f
                out_dictionery['F1'][i]=f1_list
                out_dictionery['F2'][i]=f2_list
                out_dictionery['F3'][i]=f3_list

            else:
                # out_dictionery['pitch'][i] = []
                # out_dictionery['pitch_obj'][i] = []
                out_dictionery['time_f'][i]=[]
                out_dictionery['F1'][i]=[]
                out_dictionery['F2'][i]=[]
                out_dictionery['F3'][i]=[]
            # 
            i += 1

    return out_dictionery
    



@jit( forceobj=True)    
def process_pitch_formants(ADOS_speaker,sound,window_length,time_step):
    
    out_dictionery = {'F1':{},'F2':{},'F3':{},'pitch':{} }
    
    for i in ADOS_speaker.index:
            
            snd_part = sound.extract_part(from_time=ADOS_speaker.loc[i,0],
                                                         to_time =ADOS_speaker.loc[i,1], preserve_times=True )
        
            # snd_part = sound.extract_part(from_time=d['Start_time'][0],to_time =d['End_time'][0], preserve_times=True )

            pitch = call(snd_part, "To Pitch (cc)", 0,              # time_step(s)
                         60,                                        # f0min pitch_floor(Hz)
                         15,
                         'no',
                         0.03,                                      # silence_threshold
                         0.45,                                      # Voicing_th               
                         0.01,
                         0.35,
                         0.14,
                         1600)                                      #  f0max
            # pitch = snd_part.to_pitch()
            pitch_values = pitch.selected_array['frequency'] 
            pitch_values[pitch_values==0] = np.nan
            out_dictionery['pitch'][i] = pitch_values
            
            # pre_emphasized_sound = snd_part.copy() 
            # pre_emphasized_sound.pre_emphasize() 
            
            f0min=60
            f0max=1600
            pointProcess = call(snd_part, "To PointProcess (periodic, cc)",
                                f0min,
                                f0max)
            numPoints = call(pointProcess, "Get number of points")
               
            formants = call(snd_part, "To Formant (burg)",
                            time_step,                                  # time step(s),
                            5,  
                            8000,                                       # formant ceiling(Hz),
                            window_length,                              # window length(s),
                            50)                                         # Pre-emphasis from(Hz)
            
            f1_list = []
            f2_list = []
            f3_list = []
            f4_list = []
            time_f = []
            
                # Measure formants only at glottal pulses
            for point in range(0, numPoints):
                point += 1
                t = call(pointProcess, "Get time from index", point)
                f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
                f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
                f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
                f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
                time_f.append(t)
                f1_list.append(f1)
                f2_list.append(f2)
                f3_list.append(f3)
                f4_list.append(f4)
                    
            if f1_list:
                f1_list_no_nan = [f1 for f1 in f1_list if str(f1) != 'nan']
                f2_list_no_nan = [f2 for f2 in f2_list if str(f2) != 'nan']
                f3_list_no_nan = [f3 for f3 in f3_list if str(f3) != 'nan']
                out_dictionery['F1'][i]=f1_list_no_nan
                out_dictionery['F2'][i]=f2_list_no_nan
                out_dictionery['F3'][i]=f3_list_no_nan

            else:
                out_dictionery['F1'][i]=[]
                out_dictionery['F2'][i]=[]
                out_dictionery['F3'][i]=[]
            # 
            i += 1

    return out_dictionery

@jit(forceobj=True) 
def extract_formants_pitch(ADOS_dictionery):
    
    F1 = []
    F2 = []
    F3 = []

    Pitch = []   
    for i in range(0,len(ADOS_dictionery['pitch'])):
        # print(i)
        pitch = ADOS_dictionery['pitch'][i]
        # pitch[pitch==0] = np.NaN
        pitch_no_nan = [f1 for f1 in pitch if str(f1) != 'nan']
        # pitch_interp =interp1d(t_pitch,pitch, kind='linear')(t_new) 
        Pitch.append( pitch_no_nan)

        F1.append(ADOS_dictionery['F1'][i])
        F2.append(ADOS_dictionery['F2'][i])
        F3.append(ADOS_dictionery['F3'][i])
        i+=1        
       
        
    F1_chain = np.concatenate([F1[x] for x in range(0,len(F1))])
    F2_chain = np.concatenate([F2[x] for x in range(0,len(F2))])
    F3_chain = np.concatenate([F3[x] for x in range(0,len(F3))])
    Pitch_chain = np.concatenate([Pitch[x] for x in range(0,len(Pitch))])
    
    mean_pitch = statistics.mean( Pitch_chain)           
    # calculate mean formants across pulses
    f1_mean = statistics.mean(F1_chain)
    f2_mean = statistics.mean(F2_chain)
    f3_mean = statistics.mean(F3_chain)

    f1_median = statistics.median(F1_chain)
    f2_median = statistics.median(F3_chain)
    f3_median = statistics.median(F2_chain)
    return mean_pitch,f1_mean,f2_mean,f3_mean,f1_median,f2_median,f3_median




def compute_filterbank(low_freq_mel,NFFT,nfilt,Fs):
    high_freq_mel = (2595 * np.log10(1 + (Fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    binn = np.floor((NFFT + 1) * hz_points / Fs)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 ))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(binn[m - 1])   # left
        f_m = int(binn[m])             # center
        f_m_plus = int(binn[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - binn[m - 1]) / (binn[m] - binn[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (binn[m + 1] - k) / (binn[m + 1] - binn[m])
    filter_banks = fbank 
    return filter_banks
    
def CMVN(vec, variance_normalization=False):
    """ This function is aimed to perform global cepstral mean and
        variance normalization (CMVN) on input feature vector "vec".
        The code assumes that there is one observation per row.
    Args:
        vec (array): input feature matrix
            (size:(num_observation,num_features))
        variance_normalization (bool): If the variance
            normilization should be performed or not.
    Return:
          array: The mean(or mean+variance) normalized feature vector.
    """
    eps = np.finfo(float).eps
    rows, cols = vec.shape

    # Mean calculation
    norm = np.mean(vec, axis=0)
    norm_vec = np.tile(norm, (rows, 1))

    # Mean subtraction
    mean_subtracted = vec - norm_vec

    # Variance normalization
    if variance_normalization:
        stdev = np.std(mean_subtracted, axis=0)
        stdev_vec = np.tile(stdev, (rows, 1))
        output = mean_subtracted / (stdev_vec + eps)
    else:
        output = mean_subtracted

    return output

@jit(nopython=True)
def compute_accumulated_cost_matrix_subsequence_dtw(C):
    """Given the cost matrix, compute the accumulated cost matrix for
    subsequence dynamic time warping with step sizes {(1, 0), (0, 1), (1, 1)}

    Args:
        C (np.ndarray): Cost matrix

    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N, M = C.shape
    D = np.zeros((N, M))
    D[:, 0] = np.cumsum(C[:, 0])
    D[0, :] = C[0, :]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
    return D

@jit(nopython=True)
def compute_accumulated_cost_matrix_subsequence_dtw_21(C):
    """Given the cost matrix, compute the accumulated cost matrix for
    subsequence dynamic time warping with step sizes sigma= {(1, 1), (2, 1), (1, 2)}

    Args:
        C (np.ndarray): Cost matrix

    Returns:
        D (np.ndarray): Accumulated cost matrix
    """
    N, M = C.shape
    D = np.zeros((N + 1, M + 2))
    D[0:1, :] = np.inf
    D[:, 0:2] = np.inf
# This initialization makes it possible to start at any position of the
# sequence Y without accumulating any cost
    D[1, 2:] = C[0, :]

    for n in range(1, N):
        for m in range(0, M):
            if n == 0 and m == 0:
                continue
            D[n+1, m+2] = C[n, m] + min(D[n-1+1, m-1+2], D[n-2+1, m-1+2], D[n-1+1, m-2+2])
    D = D[1:, 2:]
    return D

def mininma_from_matching_function(Delta, rho=2, tau=0.2, num=None):
    """Derives local minima positions of matching function in an iterative fashion

    Args:
        Delta (np.ndarray): Matching function
        rho (int): Parameter to exclude neighborhood of a matching position for subsequent matches (Default value = 2)
        tau (float): Threshold for maximum Delta value allowed for matches (Default value = 0.2)
        num (int): Maximum number of matches (Default value = None)

    Returns:
        pos (np.ndarray): Array of local minima
    """
    Delta_tmp = Delta.copy()
    M = len(Delta)
    pos = []
    num_pos = 0
    rho = int(rho)
    if num is None:
        num = M
    while num_pos < num and np.sum(Delta_tmp < tau) > 0:
        m = np.argmin(Delta_tmp)
        pos.append(m)
        num_pos += 1
        Delta_tmp[max(0, m - rho):min(m + rho, M)] = np.inf
    pos = np.array(pos).astype(int)
    return pos

def matches_dtw(pos, D, stepsize=2):
    """Derives matches from positions for DTW-based strategy

    Args:
        pos (np.ndarray): End positions of matches
        D (np.ndarray): Accumulated cost matrix
        stepsize (int): Parameter for step size condition (1 or 2) (Default value = 2)

    Returns:
        matches (np.ndarray): Array containing matches (start, end)
    """
    matches = np.zeros((len(pos), 2)).astype(int)
    for k in range(len(pos)):
        t = pos[k]
        matches[k, 1] = t
        if stepsize == 1:
            P = compute_optimal_warping_path_subsequence_dtw(D, m=t)
        if stepsize == 2:
            P = compute_optimal_warping_path_subsequence_dtw_21(D, m=t)
        s = P[0, 1]
        matches[k, 0] = s
    return matches

@jit(nopython=True)
def compute_optimal_warping_path_subsequence_dtw(D, m=-1):
    """Given an accumulated cost matrix, compute the warping path for
    subsequence dynamic time warping with step sizes {(1, 0), (0, 1), (1, 1)}
    Args:
        D (np.ndarray): Accumulated cost matrix
        m (int): Index to start back tracking; if set to -1, optimal m is used (Default value = -1)

    Returns:
        P (np.ndarray): Optimal warping path (array of index pairs)
    """
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = D[N - 1, :].argmin()
    P = [(n, m)]

    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-1, m]:
                cell = (n-1, m)
            else:
                cell = (n, m-1)
        P.append(cell)
        n, m = cell
    P.reverse()
    P = np.array(P)
    return P

@jit(nopython=True)
def compute_optimal_warping_path_subsequence_dtw_21(D, m=-1):
    """Given an accumulated cost matrix, compute the warping path for
    subsequence dynamic time warping with step sizes {(1, 1), (2, 1), (1, 2)}

    Args:
        D (np.ndarray): Accumulated cost matrix
        m (int): Index to start back tracking; if set to -1, optimal m is used (Default value = -1)

    Returns:
        P (np.ndarray): Optimal warping path (array of index pairs)
    """
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = D[N - 1, :].argmin()
    P = [(n, m)]

    while n > 0:
        if m == 0:
            cell = (n-1, 0)
        else:
            val = min(D[n-1, m-1], D[n-2, m-1], D[n-1, m-2])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-2, m-1]:
                cell = (n-2, m-1)
            else:
                cell = (n-1, m-2)
        P.append(cell)
        n, m = cell
    P.reverse()
    P = np.array(P)
    return P

# @jit(nopython=True)
def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
    """Normalizes the columns of a feature sequence

    Args:
        X (np.ndarray): Feature sequence
        norm (str): The norm to be applied. '1', '2', 'max' or 'z' (Default value = '2')
        threshold (float): An threshold below which the vector ``v`` used instead of normalization
            (Default value = 0.0001)
        v (float): Used instead of normalization below ``threshold``. If None, uses unit vector for given norm
            (Default value = None)

    Returns:
        X_norm (np.ndarray): Normalized feature sequence
    """
    assert norm in ['1', '2', 'max', 'z']

    K, N = X.shape
    X_norm = np.zeros((K, N))

    if norm == '1':
        if v is None:
            v = np.ones(K, dtype=np.float64) / K
        for n in range(N):
            s = np.sum(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == '2':
        if v is None:
            v = np.ones(K, dtype=np.float64) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'max':
        if v is None:
            v = np.ones(K, dtype=np.float64)
        for n in range(N):
            s = np.max(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'z':
        if v is None:
            v = np.zeros(K, dtype=np.float64)
        for n in range(N):
            mu = np.sum(X[:, n]) / K
            sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
            if sigma > threshold:
                X_norm[:, n] = (X[:, n] - mu) / sigma
            else:
                X_norm[:, n] = v

    return X_norm

def plot_matches(ax, matches, Delta, Fr=1, alpha=0.2, color='r', s_marker='o', t_marker=''):
    """Plots matches into existing axis

    Args:
        ax: Axis
        matches: Array of matches (start, end)
        Delta: Matching function
        Fr: Feature rate (Default value = 1)
        alpha: Transparency pramaeter for match visualization (Default value = 0.2)
        color: Color used to indicated matches (Default value = 'r')
        s_marker: Marker used to indicate start of matches (Default value = 'o')
        t_marker: Marker used to indicate end of matches (Default value = '')
    """
    y_min, y_max = ax.get_ylim()
    for (s, t) in matches:
        ax.plot(s/Fr, Delta[s], color=color, marker=s_marker, linestyle='None')
        ax.plot(t/Fr, Delta[t], color=color, marker=t_marker, linestyle='None')
        rect = patches.Rectangle(((s-0.5)/Fr, y_min), (t-s+1)/Fr, y_max, facecolor=color, alpha=alpha)
        ax.add_patch(rect)


def plot_signal(x, Fs=1, T_coef=None, ax=None, figsize=(6, 2), xlabel='Time (seconds)', ylabel='', title='', dpi=72,
                ylim=True, **kwargs):
    """Line plot visualization of a signal, e.g. a waveform or a novelty function.

    Args:
        x: Input signal
        Fs: Sample rate (Default value = 1)
        T_coef: Time coeffients. If None, will be computed, based on Fs. (Default value = None)
        ax: The Axes instance to plot on. If None, will create a figure and axes. (Default value = None)
        figsize: Width, height in inches (Default value = (6, 2))
        xlabel: Label for x axis (Default value = 'Time (seconds)')
        ylabel: Label for y axis (Default value = '')
        title: Title for plot (Default value = '')
        dpi: Dots per inch (Default value = 72)
        ylim: True or False (auto adjust ylim or nnot) or tuple with actual ylim (Default value = True)
        **kwargs: Keyword arguments for matplotlib.pyplot.plot

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        line: The line plot
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)
    if T_coef is None:
        T_coef = np.arange(x.shape[0]) / Fs 

    if 'color' not in kwargs:
        kwargs['color'] = 'gray'

    line = ax.plot(T_coef, x, **kwargs)

    ax.set_xlim([T_coef[0], T_coef[-1]])
    if ylim is True:
        ylim_x = x[np.isfinite(x)]
        x_min, x_max = ylim_x.min(), ylim_x.max()
        if x_max == x_min:
            x_max = x_max + 1
        ax.set_ylim([min(1.1 * x_min, 0.9 * x_min), max(1.1 * x_max, 0.9 * x_max)])
    elif ylim not in [True, False, None]:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if fig is not None:
        plt.tight_layout()

    return fig, ax, line

def plot_matrix(X, Fs=1, Fs_F=1, T_coef=None, F_coef=None, xlabel='Time (seconds)', ylabel='Frequency (Hz)',
                xlim=None, ylim=None, clim=None, title='', dpi=72,
                colorbar=True, colorbar_aspect=20.0, cbar_label='', ax=None, figsize=(6, 3), **kwargs):
    """2D raster visualization of a matrix, e.g. a spectrogram or a tempogram.

    Args:
        X: The matrix
        Fs: Sample rate for axis 1 (Default value = 1)
        Fs_F: Sample rate for axis 0 (Default value = 1)
        T_coef: Time coeffients. If None, will be computed, based on Fs. (Default value = None)
        F_coef: Frequency coeffients. If None, will be computed, based on Fs_F. (Default value = None)
        xlabel: Label for x-axis (Default value = 'Time (seconds)')
        ylabel: Label for y-axis (Default value = 'Frequency (Hz)')
        xlim: Limits for x-axis (Default value = None)
        ylim: Limits for y-axis (Default value = None)
        clim: Color limits (Default value = None)
        title: Title for plot (Default value = '')
        dpi: Dots per inch (Default value = 72)
        colorbar: Create a colorbar. (Default value = True)
        colorbar_aspect: Aspect used for colorbar, in case only a single axes is used. (Default value = 20.0)
        cbar_label: Label for colorbar (Default value = '')
        ax: Either (1.) a list of two axes (first used for matrix, second for colorbar), or (2.) a list with a single
            axes (used for matrix), or (3.) None (an axes will be created). (Default value = None)
        figsize: Width, height in inches (Default value = (6, 3))
        **kwargs: Keyword arguments for matplotlib.pyplot.imshow

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        im: The image plot
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = [ax]
    if T_coef is None:
        T_coef = np.arange(X.shape[1]) * (Fs*0.01)
    if F_coef is None:
        F_coef = np.arange(X.shape[0]) / Fs_F

    if 'extent' not in kwargs:
        x_ext1 = (T_coef[1] - T_coef[0]) / 2
        x_ext2 = (T_coef[-1] - T_coef[-2]) / 2
        y_ext1 = (F_coef[1] - F_coef[0]) / 2
        y_ext2 = (F_coef[-1] - F_coef[-2]) / 2
        kwargs['extent'] = [T_coef[0] - x_ext1, T_coef[-1] + x_ext2, F_coef[0] - y_ext1, F_coef[-1] + y_ext2]
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray_r'
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'

    im = ax[0].imshow(X, **kwargs)

    if len(ax) == 2 and colorbar:
        cbar = plt.colorbar(im, cax=ax[1])
        cbar.set_label(cbar_label)
    elif len(ax) == 2 and not colorbar:
        ax[1].set_axis_off()
    elif len(ax) == 1 and colorbar:
        plt.sca(ax[0])
        cbar = plt.colorbar(im, aspect=colorbar_aspect)
        cbar.set_label(cbar_label)

    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(title)
    if xlim is not None:
        ax[0].set_xlim(xlim)
    if ylim is not None:
        ax[0].set_ylim(ylim)
    if clim is not None:
        im.set_clim(clim)

    if fig is not None:
        plt.tight_layout()

    return fig, ax, im

def compressed_gray_cmap(alpha=5, N=256, reverse=False):
    """Create a logarithmically or exponentially compressed grayscale colormap.

    Args:
        alpha (float): The compression factor. If alpha > 0, it performs log compression (enhancing black colors).
            If alpha < 0, it performs exp compression (enhancing white colors).
            Raises an error if alpha = 0. (Default value = 5)
        N (int): The number of rgb quantization levels (usually 256 in matplotlib) (Default value = 256)
        reverse (bool): If False then "white to black", if True then "black to white" (Default value = False)

    Returns:
        color_wb (mpl.colors.LinearSegmentedColormap): The colormap
    """
    assert alpha != 0

    gray_values = np.log(1 + abs(alpha) * np.linspace(0, 1, N))
    gray_values /= gray_values.max()

    if alpha > 0:
        gray_values = 1 - gray_values
    else:
        gray_values = gray_values[::-1]

    if reverse:
        gray_values = gray_values[::-1]

    gray_values_rgb = np.repeat(gray_values.reshape(N, 1), 3, axis=1)
    color_wb = LinearSegmentedColormap.from_list('color_wb', gray_values_rgb, N=N)
    return color_wb

def matching_function_diag_multiple(X, Y, tempo_rel_set=[1], cyclic=False):
    """
    Computes diagonal matching function using multiple query strategy

    Args:
        X (np.ndarray): First sequence (K x N matrix)
        Y (np.ndarray): Second sequence (K x M matrix)
        tempo_rel_set (np.ndarray): Set of relative tempo values (scaling) (Default value = [1])
        cyclic (bool): If "True" then matching is done cyclically (Default value = False)

    Returns:
        Delta_min (np.ndarray): Matching function (obtained by from minimizing over several matching functions)
        Delta_N (np.ndarray): Query length of best match for each time position
        Delta_scale (np.ndarray): Set of matching functions (for each of the scaled versions of the query)
    """
    M = Y.shape[1]
    num_tempo = len(tempo_rel_set)
    Delta_scale = np.zeros((num_tempo, M))
    N_scale = np.zeros(num_tempo)
    for k in range(num_tempo):
        X_scale, N_scale[k] = scale_tempo_sequence(X, factor=tempo_rel_set[k])
        C_scale = 1- X_scale.T@ Y
        Delta_scale[k, :] = matching_function_diag(C_scale, cyclic=cyclic)
    Delta_min = np.min(Delta_scale, axis=0)
    Delta_argmin = np.argmin(Delta_scale, axis=0)
    Delta_N = N_scale[Delta_argmin]
    return Delta_min, Delta_N, Delta_scale

def scale_tempo_sequence(X, factor=1):
    """Scales a sequence (given as feature matrix) along time (second dimension)

    Args:
        X (np.ndarray): Feature sequences (given as K x N matrix)
        factor (float): Scaling factor (resulting in length "round(factor * N)"") (Default value = 1)

    Returns:
        X_new (np.ndarray): Scaled feature sequence
        N_new (int): Length of scaled feature sequence
    """
    N = X.shape[1]
    t = np.linspace(0, 1, num=N, endpoint=True)
    N_new = np.round(factor * N).astype(int)
    t_new = np.linspace(0, 1, num=N_new, endpoint=True)
    X_new = scipy.interpolate.interp1d(t, X, axis=1)(t_new)
    return X_new, N_new

def matching_function_diag(C, cyclic=False):
    """Computes diagonal matching function

    Args:
        C (np.ndarray): Cost matrix
        cyclic (bool): If "True" then matching is done cyclically (Default value = False)

    Returns:
        Delta (np.ndarray): Matching function
    """
    N, M = C.shape
    assert N <= M, "N <= M is required"
    Delta = C[0, :]
    for n in range(1, N):
        Delta = Delta + np.roll(C[n, :], -n)
    Delta = Delta / N
    if cyclic is False:
        Delta[M-N+1:M] = np.inf
    return Delta

def matches_diag(pos, Delta_N):
    """Derives matches from positions in the case of diagonal matching

    Args:
        pos (np.ndarray or list): Starting positions of matches
        Delta_N (int or np.ndarray or list): Length of match (a single number or a list of same length as Delta)

    Returns:
        matches (np.ndarray): Array containing matches (start, end)
    """
    matches = np.zeros((len(pos), 2)).astype(int)
    for k in range(len(pos)):
        s = pos[k]
        matches[k, 0] = s
        if isinstance(Delta_N, int):
            matches[k, 1] = s + Delta_N - 1
        else:
            matches[k, 1] = s + Delta_N[s] - 1
    return matches


def VTLN(frame, alpha=1.12):
    """Frequency warping in symmetrical manner

    Args:
        frame (np.array): one-side spectrum of signal
        alpha (float): Length of match (a single number or a list of same length as Delta) (Default value = 1.12)

    Returns:
        warpedFrame (np.array): warped one-side spectrum of signal
    """
    m = len(frame);
    omega = np.arange(m) / m * np.pi
    omega_warped = omega

    omega0 = 7/8 * np.pi

    if alpha > 1:
        omega0 = 7/(8*alpha) * np.pi
      
    # omega_warped((omega <= omega0)) = alpha .* omega((omega <= omega0));
    # omega_warped((omega > omega0)) = alpha * omega0 + ((pi - alpha * omega0)/(pi - omega0)) .* (omega((omega > omega0)) - omega0);
    omega_warped[omega <= omega0] == alpha * omega[omega <= omega0]
    omega_warped[omega > omega0] == alpha * omega0 + ((np.pi - alpha * omega0)/(np.pi - omega0)) * (omega[omega > omega0] - omega0)
    # omega_warped = omega_warped / np.pi * m
    # warpedFrame = scipy.interpolate.interp1d(np.arange(m), frame)(omega_warped)#.';%spline ,'linear','extrap'
    warpedFrame = scipy.interpolate.interp1d(omega, frame, axis=0)(omega_warped)
    return warpedFrame
    
# import numpy as np
# from scipy.interpolate import interp1d

# def interpolate_with_nans(t, y, t_new, kind='linear'):
#     """Interpolate a set of data points, preserving NaN regions from the original data.
    
#     Parameters
#     ----------
#     t : array-like
#         The time coordinates of the data points.
#     y : array-like
#         The y-coordinates of the data points.
#     t_new : array-like
#         The new time coordinates to interpolate the data onto.
#     kind : str, optional
#         The type of interpolation to use. Can be 'linear', 'nearest', 'zero', 'slinear', 'quadratic', or 'cubic'.
#         Default is 'linear'.
        
#     Returns
#     -------
#     y_interp : ndarray
#         The interpolated data, with NaN regions preserved from the original data.
#     """
#     # Find the indices of the NaN values in y
#     y = np.array(y)
#     t = np.array(t)
    
#     nan_indices = np.where(np.isnan(y))[0]
#     nan_indices = nan_indices.astype(int)
#     # Replace NaN values in y with the corresponding t-coordinate
#     t_nan = t[nan_indices]
#     y_nan = y[nan_indices]
#     y[nan_indices] = t[nan_indices]
    
#     # Interpolate the data
#     y_interp = interp1d(t, y, kind=kind)(t_new)
    
#     # Replace the interpolated values at the NaN regions with NaN
#     y_interp[np.isin(t_new, t_nan)] = np.nan
    
#     return y_interp