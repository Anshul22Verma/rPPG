import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm


import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags
from copy import deepcopy

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = len(input_signal)
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

def power2db(mag):
    """Convert power to db."""
    return 10 * np.log10(mag)

def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak

def _compute_macc(pred_signal, gt_signal):
    """Calculate maximum amplitude of cross correlation (MACC) by computing correlation at all time lags.
        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
        Returns:
            MACC(float): Maximum Amplitude of Cross-Correlation
    """
    pred = deepcopy(pred_signal)
    gt = deepcopy(gt_signal)
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    min_len = np.min((len(pred), len(gt)))
    pred = pred[:min_len]
    gt = gt[:min_len]
    lags = np.arange(0, len(pred)-1, 1)
    tlcc_list = []
    for lag in lags:
        cross_corr = np.abs(np.corrcoef(
            pred, np.roll(gt, lag))[0][1])
        tlcc_list.append(cross_corr)
    macc = max(tlcc_list)
    return macc

def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.75 Hz
        to 2.5 Hz. 

        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning 
        SNR = power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR

def calculate_metric_per_video(predictions, labels, fs=30, hr_method='FFT'):
    """Calculate video-level HR and SNR"""

    predictions = _detrend(predictions, 100)
    labels = _detrend(labels, 100)

    # bandpass filter between [0.75, 2.5] Hz
    # equals [45, 150] beats per min
    [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
    labels = scipy.signal.filtfilt(b, a, np.double(labels))
    
    macc = _compute_macc(predictions, labels)

    if hr_method == 'FFT':
        hr_pred = _calculate_fft_hr(predictions, fs=fs)
        hr_label = _calculate_fft_hr(labels, fs=fs)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    SNR = _calculate_SNR(predictions, hr_label, fs=fs)
    return hr_label, hr_pred, SNR, macc

def generate_plots(gt, predictions, result_dir: str, file_name: str):
    """
    Function to generate a Bland-Altman plot and a scatter plot with x=y line.
    
    Parameters:
    gt (array-like): Ground truth values.
    predictions (array-like): Predicted values.
    """
    # Ensure gt and predictions are numpy arrays
    gt = np.array(gt)
    predictions = np.array(predictions)

    # Bland-Altman plot
    mean_vals = (gt + predictions) / 2
    diff_vals = gt - predictions

    # Plotting Bland-Altman and Scatter plots
    plt.figure(figsize=(12, 6))

    # Bland-Altman plot
    plt.subplot(1, 2, 1)  # First subplot: Bland-Altman plot
    plt.scatter(mean_vals, diff_vals, color='blue')
    plt.axhline(np.mean(diff_vals), color='red', linestyle='--')  # Mean difference line
    plt.axhline(np.mean(diff_vals) + 1.96 * np.std(diff_vals), color='green', linestyle='--')  # Upper limit of agreement
    plt.axhline(np.mean(diff_vals) - 1.96 * np.std(diff_vals), color='green', linestyle='--')  # Lower limit of agreement
    plt.title('Bland-Altman Plot')
    plt.xlabel('Mean of GT and Prediction')
    plt.ylabel('Difference (GT - Prediction)')

    # Scatter plot with x=y line
    plt.subplot(1, 2, 2)  # Second subplot: Scatter plot
    plt.scatter(gt, predictions, color='blue')
    plt.plot([min(gt), max(gt)], [min(gt), max(gt)], color='black', linestyle='--')  # x=y line
    plt.title('Ground Truth vs Prediction')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, file_name))
    plt.close()


def calculate_metrics(predictions, labels, fs, result_dir):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    SNR_fft_all = list()
    
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_peak_all = list()
    
    MACC_all = list()
    print("Calculating metrics!")

    predictions = [prediction.squeeze().tolist() for prediction in predictions]
    labels = [label.tolist() for label in labels]
    print(len(predictions))

    for index in tqdm(range(len(predictions)), ncols=80):
        prediction = predictions[index]  # shape of the chunk
        label = labels[index]  # shape of the chunk
        
        gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(predictions=prediction, labels=label,
                                                                          fs=fs, hr_method='Peak')
        gt_hr_peak_all.append(gt_hr_peak)
        predict_hr_peak_all.append(pred_hr_peak)
        SNR_peak_all.append(SNR)
        MACC_all.append(macc)

        gt_hr_fft, pred_hr_fft, SNR, _ = calculate_metric_per_video(predictions=prediction, labels=label, 
                                                                       fs=fs, hr_method='FFT')
        gt_hr_fft_all.append(gt_hr_fft)
        predict_hr_fft_all.append(pred_hr_fft)
        SNR_fft_all.append(SNR)
    
    gt_hr_fft_all = np.array(gt_hr_fft_all)
    predict_hr_fft_all = np.array(predict_hr_fft_all)
    SNR_fft_all = np.array(SNR_fft_all)
    MACC_all = np.array(MACC_all)
    gt_hr_peak_all = np.array(gt_hr_peak_all)
    predict_hr_peak_all = np.array(predict_hr_peak_all)
    SNR_peak_all = np.array(SNR_peak_all)

    num_test_samples = len(predict_hr_fft_all)
    
    # Compute MAE
    MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
    standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
    print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
    
    MAE_Peak = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
    standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
    print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_Peak, standard_error))
    
    # Compute RMSE
    squared_errors = np.square(predict_hr_fft_all - gt_hr_fft_all)
    RMSE_FFT = np.sqrt(np.mean(squared_errors))
    standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
    print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))

    squared_errors = np.square(predict_hr_peak_all - gt_hr_peak_all)
    RMSE_Peak = np.sqrt(np.mean(squared_errors))
    standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
    print("Peak RMSE (Peak Label): {0} +/- {1}".format(RMSE_Peak, standard_error))

    # Compute MAPE
    MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
    standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
    print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))

    MAPE_Peak = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
    standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
    print("Peak MAPE (Peak Label): {0} +/- {1}".format(MAPE_Peak, standard_error))

    # Compute Pearson Correlation            
    Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
    correlation_coefficient = Pearson_FFT[0][1]
    standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
    print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
    
    Pearson_Peak = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
    correlation_coefficient = Pearson_Peak[0][1]
    standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
    print("Peak Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))

    # Compute SNR
    SNR_FFT = np.mean(SNR_fft_all)
    standard_error = np.std(SNR_fft_all) / np.sqrt(num_test_samples)
    print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))

    SNR_Peak = np.mean(SNR_peak_all)
    standard_error = np.std(SNR_peak_all) / np.sqrt(num_test_samples)
    print("Peak SNR (Peak Label): {0} +/- {1} (dB)".format(SNR_Peak, standard_error))
    

    # Compute MACC
    MACC_avg = np.mean(MACC_all)
    standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
    print("MACC: {0} +/- {1}".format(MACC_avg, standard_error))

    # generate BlandAltman Plot and line plot prediction vs ground truth
    # for FFT
    print(len(gt_hr_fft_all))
    print(len(predict_hr_fft_all))
    generate_plots(gt_hr_fft_all, predict_hr_fft_all, result_dir=result_dir, file_name=f"FFT_plots.png")
    # for Peak
    generate_plots(gt_hr_peak_all, predict_hr_peak_all, result_dir=result_dir, file_name=f"Peak_plots.png")

    # Save results to a CSV file
    results = {
        "MAE_FFT": MAE_FFT,
        "MAE_Peak": MAE_Peak,
        "RMSE_FFT": RMSE_FFT,
        "RMSE_Peak": RMSE_Peak,
        "MAPE_FFT": MAPE_FFT,
        "MAPE_Peak": MAPE_Peak,
        "Pearson_FFT": correlation_coefficient,
        "Pearson_Peak": correlation_coefficient
    }
    
    df_results = pd.DataFrame(results, index=[0])
    result_file = f'{result_dir}/metrics_results.csv'
    df_results.to_csv(result_file, index=False)
