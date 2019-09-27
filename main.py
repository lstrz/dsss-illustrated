import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import PRN

dofig = True
savefig = False
showfig = False


# Turns 0 1 0 1 into digits[0] digits[1] digits[0] digits[1]
def binarize(xs, digits):
    return np.array([digits[0] if x <= 0 else digits[1] for x in xs])


def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits][1:])
    return result


def frombits(bits):
    chars = []
    for b in range(int(len(bits) / 7)):
        byte = bits[b * 7:(b + 1) * 7]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


def nextpow2(x):
    return int(np.ceil(np.log2(np.abs(x))))


def freq_shift(x, f_shift, dt):
    N_orig = len(x)
    N_padded = 2 ** nextpow2(N_orig)
    t = np.arange(0, N_padded)
    return (sp.hilbert(np.hstack((x, np.zeros(N_padded - N_orig, x.dtype)))) * np.exp(2j * np.pi * f_shift * dt * t))[
           :N_orig].real


plot_counter = 0


def plot_signal_and_spectrum(signal_td, t, pos_start_td, pos_end_td, pos_start_fd, pos_end_fd, title_td, xlabel_td,
                             ylabel_td, title_fd, xlabel_fd, ylabel_fd, title_window, savefig):
    global plot_counter
    plot_counter += 1

    signal_fd = np.abs(np.fft.rfft(signal_td)) ** 2
    dt = t[1] - t[0]
    signal_frequencies = np.fft.rfftfreq(signal_td.size, dt)
    df = signal_frequencies[1] - signal_frequencies[0]

    pos_start_td = int(pos_start_td / dt)
    if pos_end_td != -1:
        pos_end_td = int(pos_end_td / dt)
    pos_start_fd = int(pos_start_fd / df)
    if pos_end_fd != -1:
        pos_end_fd = int(pos_end_fd / df)

    fig = plt.figure(plot_counter)
    splt = plt.subplot(211)
    plt.plot(t[pos_start_td:pos_end_td], signal_td[pos_start_td:pos_end_td], lw=0.5)
    splt.set_title(title_td)
    splt.set_xlabel(xlabel_td)
    splt.set_ylabel(ylabel_td)

    splt = plt.subplot(212)
    plt.semilogy(signal_frequencies[pos_start_fd:pos_end_fd], signal_fd[pos_start_fd:pos_end_fd], lw=0.2)
    splt.set_title(title_fd)
    splt.set_xlabel(xlabel_fd)
    splt.set_ylabel(ylabel_fd)

    fig.canvas.set_window_title(str(plot_counter) + " " + title_window)
    if savefig:
        default_size = fig.get_size_inches()
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(plot_counter) + "-" + title_window.replace(" ", "-") + ".svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(plot_counter) + "-" + title_window.replace(" ", "-") + ".png", bbox_inches='tight')
        print(title_window + " saved.")


def plot_signal_and_two_spectrums(signal_td, t, pos_start_td, pos_end_td, pos_start_fd1, pos_end_fd1, pos_start_fd2,
                                  pos_end_fd2, title_td, xlabel_td, ylabel_td, title_fd1, xlabel_fd1, ylabel_fd1,
                                  title_fd2, xlabel_fd2, ylabel_fd2, title_window, savefig):
    global plot_counter
    plot_counter += 1

    signal_fd = np.abs(np.fft.rfft(signal_td)) ** 2
    dt = t[1] - t[0]
    signal_frequencies = np.fft.rfftfreq(signal_td.size, dt)
    df = signal_frequencies[1] - signal_frequencies[0]

    pos_start_td = int(pos_start_td / dt)
    if pos_end_td != -1:
        pos_end_td = int(pos_end_td / dt)
    pos_start_fd1 = int(pos_start_fd1 / df)
    if pos_end_fd1 != -1:
        pos_end_fd1 = int(pos_end_fd1 / df)
    pos_start_fd2 = int(pos_start_fd2 / df)
    if pos_end_fd2 != -1:
        pos_end_fd2 = int(pos_end_fd2 / df)

    fig = plt.figure(plot_counter)
    splt = plt.subplot(211)
    plt.plot(t[pos_start_td:pos_end_td], signal_td[pos_start_td:pos_end_td], lw=0.5)
    splt.set_title(title_td)
    splt.set_xlabel(xlabel_td)
    splt.set_ylabel(ylabel_td)

    splt = plt.subplot(223)
    plt.semilogy(signal_frequencies[pos_start_fd1:pos_end_fd1], signal_fd[pos_start_fd1:pos_end_fd1], lw=0.2)
    splt.set_title(title_fd1)
    splt.set_xlabel(xlabel_fd1)
    splt.set_ylabel(ylabel_fd1)

    splt = plt.subplot(224)
    plt.semilogy(signal_frequencies[pos_start_fd2:pos_end_fd2], signal_fd[pos_start_fd2:pos_end_fd2], lw=0.2)
    splt.set_title(title_fd2)
    splt.set_xlabel(xlabel_fd2)
    splt.set_ylabel(ylabel_fd2)

    fig.canvas.set_window_title(str(plot_counter) + " " + title_window)
    if savefig:
        default_size = fig.get_size_inches()
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(plot_counter) + "-" + title_window.replace(" ", "-") + ".svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(plot_counter) + "-" + title_window.replace(" ", "-") + ".png", bbox_inches='tight')
        print(title_window + " saved.")


def plot_two_signals(signal1, t1, signal2, t2, pos_start1, pos_end1, pos_start2, pos_end2, title1, xlabel1, ylabel1,
                     title2, xlabel2, ylabel2, title_window, savefig):
    global plot_counter
    plot_counter += 1

    pos_start1 = int(pos_start1)
    pos_end1 = int(pos_end1)
    pos_start2 = int(pos_start2)
    pos_end2 = int(pos_end2)

    fig = plt.figure(plot_counter)
    splt = plt.subplot(211)
    if t1 is None:
        plt.plot(signal1[pos_start1:pos_end1], lw=0.5)
    else:
        plt.plot(t1[pos_start1:pos_end1], signal1[pos_start1:pos_end1], lw=0.5)
    splt.set_title(title1)
    splt.set_xlabel(xlabel1)
    splt.set_ylabel(ylabel1)

    splt = plt.subplot(212)
    if t2 is None:
        plt.plot(signal2[pos_start2:pos_end2], lw=0.5)
    else:
        plt.plot(t2[pos_start2:pos_end2], signal2[pos_start2:pos_end2], lw=0.5)
    splt.set_title(title2)
    splt.set_xlabel(xlabel2)
    splt.set_ylabel(ylabel2)

    fig.canvas.set_window_title(str(plot_counter) + " " + title_window)
    if savefig:
        default_size = fig.get_size_inches()
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(plot_counter) + "-" + title_window.replace(" ", "-") + ".svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(plot_counter) + "-" + title_window.replace(" ", "-") + ".png", bbox_inches='tight')
        print(title_window + " saved.")


def plot_four_signals(signal1, t1, signal2, t2, signal3, t3, signal4, t4, pos_start1, pos_end1, pos_start2, pos_end2,
                      pos_start3, pos_end3, pos_start4, pos_end4, title1, xlabel1, ylabel1, title2, xlabel2, ylabel2,
                      title3, xlabel3, ylabel3, title4, xlabel4, ylabel4, title_window, savefig):
    global plot_counter
    plot_counter += 1

    pos_start1 = int(pos_start1)
    pos_end1 = int(pos_end1)
    pos_start2 = int(pos_start2)
    pos_end2 = int(pos_end2)
    pos_start3 = int(pos_start3)
    pos_end3 = int(pos_end3)
    pos_start4 = int(pos_start4)
    pos_end4 = int(pos_end4)

    fig = plt.figure(plot_counter)
    splt = plt.subplot(221)
    if t1 is None:
        plt.plot(signal1[pos_start1:pos_end1], lw=0.5)
    else:
        plt.plot(t1[pos_start1:pos_end1], signal1[pos_start1:pos_end1], lw=0.5)
    splt.set_title(title1)
    splt.set_xlabel(xlabel1)
    splt.set_ylabel(ylabel1)

    splt = plt.subplot(222)
    if t2 is None:
        plt.plot(signal2[pos_start2:pos_end2], lw=0.5)
    else:
        plt.plot(t2[pos_start2:pos_end2], signal2[pos_start2:pos_end2], lw=0.5)
    splt.set_title(title2)
    splt.set_xlabel(xlabel2)
    splt.set_ylabel(ylabel2)

    splt = plt.subplot(223)
    if t3 is None:
        plt.plot(signal3[pos_start3:pos_end3], lw=0.5)
    else:
        plt.plot(t3[pos_start3:pos_end3], signal3[pos_start3:pos_end3], lw=0.5)
    splt.set_title(title3)
    splt.set_xlabel(xlabel3)
    splt.set_ylabel(ylabel3)

    splt = plt.subplot(224)
    if t4 is None:
        plt.plot(signal4[pos_start4:pos_end4], lw=0.5)
    else:
        plt.plot(t4[pos_start4:pos_end4], signal4[pos_start4:pos_end4], lw=0.5)
    splt.set_title(title4)
    splt.set_xlabel(xlabel4)
    splt.set_ylabel(ylabel4)

    fig.canvas.set_window_title(str(plot_counter) + " " + title_window)
    if savefig:
        default_size = fig.get_size_inches()
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(plot_counter) + "-" + title_window.replace(" ", "-") + ".svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(plot_counter) + "-" + title_window.replace(" ", "-") + ".png", bbox_inches='tight')
        print(title_window + " saved.")


# Frequencies
base_frequency = 10.23e6
reference_frequency = 16.368e6

coarse_acquisition_code_frequency = base_frequency / 10
message_frequency = 500

carrier_frequency_multiplier = 7
carrier_frequency = base_frequency * carrier_frequency_multiplier

mixer_frequency_multiplier = int(carrier_frequency / reference_frequency)
mixer_frequency = reference_frequency * mixer_frequency_multiplier

intermediate_frequency = carrier_frequency - mixer_frequency
filter_bandwidth = 2.5e6
filter_lo_frequency = intermediate_frequency - filter_bandwidth / 2
filter_hi_frequency = intermediate_frequency + filter_bandwidth / 2
filter_order = 4

samples_per_carrier_period = 5
sampling_frequency = filter_hi_frequency * 4

# Time
message_code = np.array(tobits("Test"))
time_start = 0.0
time_end = 1.0 / message_frequency * message_code.size
time_duration = time_end - time_start
coarse_acquisition_code_period = 1e-3
dt = 1 / carrier_frequency / samples_per_carrier_period
time_steps = time_duration / dt
t = np.linspace(time_start, time_end, int(time_steps))

# Wave
amplitude = 1
phase = 0
noise_stddev = 40  # 40 is a CN0 of ~36.5; 27.2 is a CN0 of 40; 8.6 is a CN0 of 50
noise_offset = 0
satellite_vehicle_number = 10
satellite_vehicle_number_bad = 15

# Print info
print("Base frequency: ", base_frequency)
print("Carrier frequency: ", carrier_frequency)
print("CA code frequency: ", coarse_acquisition_code_frequency)
print("Message frequency: ", message_frequency)
print("Reference frequency: ", reference_frequency)
print("Reference multiplier: ", mixer_frequency_multiplier)
print("Mixer frequency: ", mixer_frequency)
print("Intermediate frequency: ", intermediate_frequency)
print("Filter bandwidth: ", filter_bandwidth)
print("Sampling frequency: ", sampling_frequency)

# Carrier
carrier_signal = amplitude * np.sin(2 * np.pi * carrier_frequency * t + phase)
print("Carrier done.")

# CA code
original_binarized_coarse_acquisition_code = np.array(PRN.PRN(satellite_vehicle_number))
original_binarized_coarse_acquisition_code_bad = np.array(PRN.PRN(satellite_vehicle_number_bad))
coarse_acquisition_code = np.tile(original_binarized_coarse_acquisition_code,
                                  int(time_duration / coarse_acquisition_code_period))
coarse_acquisition_code = np.repeat(coarse_acquisition_code, t.size / coarse_acquisition_code.size)
coarse_acquisition_code_signal = binarize(coarse_acquisition_code, [-1, 1])
print("CA code done.")

# Message
message_signal = binarize(np.repeat(message_code, t.size / message_code.size), [-1, 1])
print("Message done.")

# Output signal
output_signal = message_signal * carrier_signal * coarse_acquisition_code_signal
print("Output done.")

# Input signal
input_signal = output_signal + np.random.normal(noise_offset, noise_stddev, output_signal.size)
print("Input done.")

# Mixer signal
mixer_signal = amplitude * np.sin(2 * np.pi * mixer_frequency * t + phase)
print("Mixer done.")

# Mixed signal
mixed_signal = input_signal * mixer_signal
print("Mixed done.")

# Filtered signal
mixed_frequencies = np.fft.rfftfreq(mixed_signal.size, dt)
filter_nyquist_frequency = mixed_frequencies[-1]
z, p, k = sp.butter(filter_order,
                    [filter_lo_frequency / filter_nyquist_frequency, filter_hi_frequency / filter_nyquist_frequency],
                    btype='bandpass', output='zpk')
sos = sp.zpk2sos(z, p, k)
filtered_signal = sp.sosfiltfilt(sos, mixed_signal)
print("Filtered done.")

# Resampling
sampled_signal, sampled_t = sp.resample(filtered_signal, int(sampling_frequency * time_duration), t)
sampled_dt = time_duration / sampled_t.size
print("Resampled done.")

# Binarizing
binarized_sampled_signal = binarize(sampled_signal, [-1, 1])
binarized_coarse_acquisition_code = np.tile(original_binarized_coarse_acquisition_code,
                                            int(time_duration / coarse_acquisition_code_period))
binarized_coarse_acquisition_code = np.repeat(binarized_coarse_acquisition_code,
                                              sampled_t.size / binarized_coarse_acquisition_code.size)
binarized_coarse_acquisition_code = sp.resample(binarized_coarse_acquisition_code,
                                                int(sampling_frequency * time_duration))
binarized_coarse_acquisition_code_signal = binarize(binarized_coarse_acquisition_code, [-1, 1])
print("Binarized done.")

# Digitizing bad PRN
binarized_coarse_acquisition_code_bad = np.tile(original_binarized_coarse_acquisition_code_bad,
                                                int(time_duration / coarse_acquisition_code_period))
binarized_coarse_acquisition_code_bad = np.repeat(binarized_coarse_acquisition_code_bad,
                                                  sampled_t.size / binarized_coarse_acquisition_code_bad.size)
binarized_coarse_acquisition_code_bad = sp.resample(binarized_coarse_acquisition_code_bad,
                                                    int(sampling_frequency * time_duration))
binarized_coarse_acquisition_code_signal_bad = binarize(binarized_coarse_acquisition_code_bad, [-1, 1])

# Reference carrier
reference_carrier_signal = sp.resample(sp.sosfiltfilt(sos, carrier_signal * mixer_signal),
                                       int(sampling_frequency * time_duration))
binarized_reference_carrier_signal = binarize(reference_carrier_signal, [-1, 1])
print("Reference carrier done.")

# Removing carrier
binarized_no_carrier_signal = binarized_sampled_signal * binarized_reference_carrier_signal
print("Reference removal done.")

# Demodulated
demodulated_signal = binarized_no_carrier_signal * binarized_coarse_acquisition_code_signal
demodulated_signal_bad = binarized_no_carrier_signal * binarized_coarse_acquisition_code_signal_bad
print("Demodulation done.")

# Decoding
decoded_message_code = binarize(np.average(demodulated_signal.reshape((int(time_duration * message_frequency), -1)), 1), [0, 1])
decoded_message_code_bad = binarize(np.average(demodulated_signal_bad.reshape((int(time_duration * message_frequency), -1)), 1), [0, 1])
decoded_message = frombits(decoded_message_code)
decoded_message_bad = frombits(decoded_message_code_bad)
print("Decoding/integration done.")
print("Decoded message: " + decoded_message)
print("Decoded message bad: " + decoded_message_bad)

# Plot
if dofig:
    pos_start_td = 2.4e-6 - 0.7e-6
    pos_end_td = 2.4e-6 + 0.7e-6
    plot_signal_and_spectrum(carrier_signal, t, pos_start_td, pos_end_td,
                             carrier_frequency - coarse_acquisition_code_frequency,
                             carrier_frequency + coarse_acquisition_code_frequency, "Carrier signal time domain",
                             "Time [s]",
                             "Amplitude [V]", "Carrier signal frequency domain", "Frequency [Hz]", "Power [W]",
                             "Carrier signal", savefig)

    plot_signal_and_spectrum(coarse_acquisition_code_signal, t, pos_start_td, pos_end_td, 0,
                             coarse_acquisition_code_frequency * 10,
                             "PRN code time domain", "Time [s]", "Amplitude [V]", "PRN code frequency domain",
                             "Frequency [Hz]", "Power [W]",
                             "Coarse acquisition code signal", savefig)

    plot_signal_and_spectrum(message_signal, t, 0, -1, 0, message_frequency * 4, "Message signal time domain",
                             "Time [s]",
                             "Amplitude [V]",
                             "Message signal frequency domain", "Frequency [Hz]", "Power [W]", "Message signal",
                             savefig)

    plot_signal_and_spectrum(output_signal, t, pos_start_td, pos_end_td,
                             carrier_frequency - coarse_acquisition_code_frequency * 10,
                             carrier_frequency + coarse_acquisition_code_frequency * 10, "Output signal time domain",
                             "Time [s]",
                             "Amplitude [V]", "Output signal frequency domain", "Frequency [Hz]", "Power [W]",
                             "Output signal", savefig)

    plot_signal_and_spectrum(input_signal, t, pos_start_td, pos_end_td,
                             carrier_frequency - coarse_acquisition_code_frequency * 10,
                             carrier_frequency + coarse_acquisition_code_frequency * 10, "Input signal time domain",
                             "Time [s]",
                             "Amplitude [V]", "Input signal frequency domain", "Frequency [Hz]", "Power [W]",
                             "Input signal", savefig)

    plot_signal_and_two_spectrums(mixed_signal, t, pos_start_td, pos_end_td, 0, intermediate_frequency * 2,
                                  intermediate_frequency + 2 * mixer_frequency - intermediate_frequency,
                                  intermediate_frequency + 2 * mixer_frequency + intermediate_frequency,
                                  "Mixed signal time domain", "Time [s]",
                                  "Amplitude [V]", "Mixed signal frequency domain, low part", "Frequency [Hz]",
                                  "Power [W]",
                                  "Mixed signal frequency domain, high part", "Frequency [Hz]", "Power [W]",
                                  "Mixed input signal", savefig)

    plot_signal_and_spectrum(filtered_signal, t, pos_start_td, pos_end_td, 0, sampling_frequency,
                             "Filtered signal time domain", "Time [s]",
                             "Amplitude [V]", "Filtered signal frequency domain", "Frequency [Hz]", "Power [W]",
                             "Filtered input signal",
                             savefig)

    plot_signal_and_spectrum(sampled_signal, sampled_t, pos_start_td, pos_end_td, 0, -1, "Sampled signal time domain",
                             "Time [s]",
                             "Amplitude [V]", "Sampled signal frequency domain", "Frequency [Hz]", "Power [W]",
                             "Sampled input signal",
                             savefig)

    plot_signal_and_spectrum(reference_carrier_signal, sampled_t, pos_start_td, pos_end_td,
                             intermediate_frequency - coarse_acquisition_code_frequency,
                             intermediate_frequency + coarse_acquisition_code_frequency,
                             "Reference carrier signal time domain", "Time [s]",
                             "Amplitude [V]", "Reference carrier signal frequency domain", "Frequency [Hz]",
                             "Power [W]",
                             "Reference carrier signal", savefig)

    plot_two_signals(demodulated_signal, sampled_t, decoded_message_code,
                     sampled_t[::int(sampled_t.size / decoded_message_code.size)],
                     0, 1 / message_frequency / (sampled_t[1] - sampled_t[0]), 0, -1,
                     "One bit of demodulated and digitized signal time domain", "Time [s]", "Amplitude [V]",
                     "Decoded message time domain", "Time [s]", "Bit value", "Decoded message", savefig)

    plot_two_signals(demodulated_signal_bad, sampled_t, decoded_message_code_bad,
                     sampled_t[::int(sampled_t.size / decoded_message_code_bad.size)],
                     0, 1 / message_frequency / (sampled_t[1] - sampled_t[0]), 0, -1,
                     "One bit of demodulated and digitized bad signal time domain", "Time [s]", "Amplitude [V]",
                     "Decoded bad message time domain", "Time [s]", "Bit value", "Decoded bad message", savefig)

    if showfig:
        plt.show()
