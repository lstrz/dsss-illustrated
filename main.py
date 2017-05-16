import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import PRN

dofig = True
savefig = True
showfig = False


# Turns 0 1 0 1 into -1 1 -1 1
def signalize(xs):
    return np.array([-1 if x <= 0 else 1 for x in xs])


# Turns -1 1 -1 1 into 0 1 0 1
def digitize(xs):
    return np.array([0 if x <= 0 else 1 for x in xs])


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

doppler_shift = 400

# Time
message_code = np.array(tobits("Te"))
time_start = 0
time_end = 1 / message_frequency * message_code.size
time_duration = time_end - time_start
coarse_acquisition_code_period = 1e-3
dt = 1 / carrier_frequency / samples_per_carrier_period
time_steps = time_duration / dt
t = np.linspace(time_start, time_end, int(time_steps))

# Wave
amplitude = 1
phase = 0
noise_stddev = 40  # 27.2 is a CN0 of 40; 8.6 is a CN0 of 50
noise_offset = 0
satellite_vehicle_number = 10

# Print info
print("Base frequency: ", base_frequency)
print("Carrier frequency: ", carrier_frequency)
print("CA code frequency: ", coarse_acquisition_code_frequency)
print("Message frequency: ", message_frequency)
print("Reference frequency: ", reference_frequency)
print("Reference multiplier: ", mixer_frequency_multiplier)
print("Intermediate frequency: ", intermediate_frequency)
print("Filter bandwidth: ", filter_bandwidth)
print("Sampling frequency: ", sampling_frequency)

# Carrier
carrier_signal = amplitude * np.sin(2 * np.pi * carrier_frequency * t + phase)
carrier_spectrum = np.abs(np.fft.rfft(carrier_signal)) ** 2
carrier_spectrum_frequencies = np.fft.rfftfreq(carrier_signal.size, dt)
print("Carrier done.")

# CA code
original_digitized_coarse_acquisition_code = np.array(PRN.PRN(satellite_vehicle_number))
coarse_acquisition_code = np.tile(original_digitized_coarse_acquisition_code,
                                  int(time_duration / coarse_acquisition_code_period))
coarse_acquisition_code = np.repeat(coarse_acquisition_code, t.size / coarse_acquisition_code.size)
coarse_acquisition_code_signal = signalize(coarse_acquisition_code)
coarse_acquisition_code_spectrum = np.abs(np.fft.rfft(coarse_acquisition_code_signal)) ** 2
coarse_acquisition_code_frequencies = np.fft.rfftfreq(coarse_acquisition_code_signal.size, dt)
print("CA code done.")

# Message
message_signal = signalize(np.repeat(message_code, t.size / message_code.size))
message_spectrum = np.abs(np.fft.rfft(message_signal)) ** 2
message_frequencies = np.fft.rfftfreq(message_signal.size, dt)
print("Message done.")

# Output signal
output_signal = carrier_signal * message_signal * coarse_acquisition_code_signal
output_spectrum = np.abs(np.fft.rfft(output_signal)) ** 2
output_frequencies = np.fft.rfftfreq(output_signal.size, dt)
print("Output done.")

# Input signal
input_signal = freq_shift(output_signal, doppler_shift, dt) + np.random.normal(noise_offset, noise_stddev,
                                                                               output_signal.size)
input_spectrum = np.abs(np.fft.rfft(input_signal)) ** 2
input_frequencies = np.fft.rfftfreq(input_signal.size, dt)
print("Input done.")

# Mixer signal
mixer_signal = amplitude * np.sin(2 * np.pi * mixer_frequency * t + phase)
mixer_spectrum = np.abs(np.fft.rfft(mixer_signal)) ** 2
mixer_frequencies = np.fft.rfftfreq(mixer_signal.size, dt)
print("Mixer done.")

# Mixed signal
mixed_signal = input_signal * mixer_signal
mixed_spectrum = np.abs(np.fft.rfft(mixed_signal)) ** 2
mixed_frequencies = np.fft.rfftfreq(mixed_signal.size, dt)
print("Mixed done.")

# Filtered signal
filter_nyquist_frequency = mixed_frequencies[-1]
z, p, k = sp.butter(filter_order,
                    [filter_lo_frequency / filter_nyquist_frequency, filter_hi_frequency / filter_nyquist_frequency],
                    btype='bandpass', output='zpk')
sos = sp.zpk2sos(z, p, k)
filtered_signal = sp.sosfiltfilt(sos, mixed_signal)
filtered_spectrum = np.abs(np.fft.rfft(filtered_signal)) ** 2
filtered_frequencies = np.fft.rfftfreq(filtered_signal.size, dt)
print("Filtered done.")

# Sampling
sampled_signal, sampled_t = sp.resample(filtered_signal, int(sampling_frequency * time_duration), t)
sampled_dt = time_duration / sampled_t.size
sampled_spectrum = np.abs(np.fft.rfft(sampled_signal)) ** 2
sampled_frequencies = np.fft.rfftfreq(sampled_signal.size, sampled_dt)
print("Sampled done.")

# Reference carrier
reference_carrier_signal = sp.resample(sp.sosfiltfilt(sos, carrier_signal * mixer_signal),
                                       int(sampling_frequency * time_duration))
reference_carrier_spectrum = np.abs(np.fft.rfft(reference_carrier_signal)) ** 2
reference_carrier_frequencies = np.fft.rfftfreq(reference_carrier_signal.size, sampled_dt)
print("Reference carrier done.")

# Digitizing
digitized_sampled_signal = signalize(sampled_signal)
digitized_reference_carrier_signal = signalize(reference_carrier_signal)
digitized_coarse_acquisition_code = np.tile(original_digitized_coarse_acquisition_code,
                                            int(time_duration / coarse_acquisition_code_period))
digitized_coarse_acquisition_code = np.repeat(digitized_coarse_acquisition_code,
                                              sampled_t.size / digitized_coarse_acquisition_code.size)
digitized_coarse_acquisition_code = sp.resample(digitized_coarse_acquisition_code,
                                                int(sampling_frequency * time_duration))
digitized_coarse_acquisition_code_signal = signalize(digitized_coarse_acquisition_code)
print("Digitized done.")

# Acquisition
digitized_coarse_acquisition_correlation_code = np.repeat(original_digitized_coarse_acquisition_code, int(
    sampling_frequency * coarse_acquisition_code_period / original_digitized_coarse_acquisition_code.size))
digitized_coarse_acquisition_correlation_code = sp.resample(digitized_coarse_acquisition_correlation_code,
                                                            int(sampling_frequency * coarse_acquisition_code_period))
digitized_coarse_acquisition_correlation_code = signalize(digitized_coarse_acquisition_correlation_code)
acquisition_signal_good = np.abs(sp.correlate(digitized_sampled_signal * digitized_reference_carrier_signal,
                                              digitized_coarse_acquisition_correlation_code))

original_digitized_coarse_acquisition_code = np.array(PRN.PRN(satellite_vehicle_number - 1))
digitized_coarse_acquisition_correlation_code = np.repeat(original_digitized_coarse_acquisition_code, int(
    sampling_frequency * coarse_acquisition_code_period / original_digitized_coarse_acquisition_code.size))
digitized_coarse_acquisition_correlation_code = sp.resample(digitized_coarse_acquisition_correlation_code,
                                                            int(sampling_frequency * coarse_acquisition_code_period))
digitized_coarse_acquisition_correlation_code = signalize(digitized_coarse_acquisition_correlation_code)
acquisition_signal_bad = np.abs(sp.correlate(digitized_sampled_signal * digitized_reference_carrier_signal,
                                             digitized_coarse_acquisition_correlation_code))

# Demodulated
demodulated_signal = digitized_sampled_signal * digitized_reference_carrier_signal * digitized_coarse_acquisition_code_signal
demodulated_spectrum = np.abs(np.fft.rfft(demodulated_signal)) ** 2
demodulated_frequencies = np.fft.rfftfreq(demodulated_signal.size, sampled_dt)
print("Demodulation done.")

decoded_message_code = digitize(np.average(demodulated_signal.reshape((int(time_duration * message_frequency), -1)), 1))
decoded_message = frombits(decoded_message_code)
print("Decoding/integration done.")
print(decoded_message)

# Plot
if dofig:
    i = 0
    i += 1
    pos1_td = int(2.4e-6 / dt) - int(0.7e-6 / dt)
    pos2_td = int(2.4e-6 / dt) + int(0.7e-6 / dt)
    df = carrier_spectrum_frequencies[-1] / carrier_spectrum_frequencies.size
    pos1_fd = int(carrier_frequency / df) - int(coarse_acquisition_code_frequency / df)
    pos2_fd = int(carrier_frequency / df) + int(coarse_acquisition_code_frequency / df)
    fig = plt.figure(i)
    splt = plt.subplot(211)
    plt.plot(t[pos1_td:pos2_td], carrier_signal[pos1_td:pos2_td], lw=0.3)
    splt.set_title("Carrier signal time domain")
    splt.set_xlabel("Time [s]")
    splt.set_ylabel("Amplitude [V]")
    splt = plt.subplot(212)
    plt.semilogy(carrier_spectrum_frequencies[pos1_fd:pos2_fd], carrier_spectrum[pos1_fd:pos2_fd], lw=0.2)
    splt.set_title("Carrier signal frequency domain")
    splt.set_xlabel("Frequency [Hz]")
    splt.set_ylabel("Power [W]")
    fig.canvas.set_window_title(str(i) + " Carrier signal")
    default_size = fig.get_size_inches()
    if savefig:
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(i) + "-carrier-signal.svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(i) + "-carrier-signal.png", bbox_inches='tight')
        print("Carrier signal saved.")

    i += 1
    df = coarse_acquisition_code_frequencies[-1] / coarse_acquisition_code_frequencies.size
    pos1_fd = 0
    pos2_fd = int(coarse_acquisition_code_frequency * 10 / df)
    fig = plt.figure(i)
    splt = plt.subplot(211)
    plt.plot(t[pos1_td:pos2_td], coarse_acquisition_code_signal[pos1_td:pos2_td], lw=0.3)
    splt.set_title("PRN code time domain")
    splt.set_xlabel("Time [s]")
    splt.set_ylabel("Amplitude [V]")
    splt = plt.subplot(212)
    plt.semilogy(coarse_acquisition_code_frequencies[pos1_fd:pos2_fd:4],
                 coarse_acquisition_code_spectrum[pos1_fd:pos2_fd:4],
                 lw=0.2)
    splt.set_title("PRN code frequency domain")
    splt.set_xlabel("Frequency [Hz]")
    splt.set_ylabel("Power [W]")
    fig.canvas.set_window_title(str(i) + " Coarse acquisition code signal")
    if savefig:
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(i) + "-coarse-acquisition-code-signal.svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(i) + "-coarse-acquisition-code-signal.png", bbox_inches='tight')
        print("CA code signal saved.")

    i += 1
    df = message_frequencies[-1] / message_frequencies.size
    pos1_fd = 0
    pos2_fd = int(message_frequency * 4 / df)
    fig = plt.figure(i)
    splt = plt.subplot(211)
    plt.plot(t, message_signal, lw=0.3)
    splt.set_title("Message signal time domain")
    splt.set_xlabel("Time [s]")
    splt.set_ylabel("Amplitude [V]")
    splt = plt.subplot(212)
    plt.semilogy(message_frequencies[pos1_fd:pos2_fd], message_spectrum[pos1_fd:pos2_fd], lw=0.2)
    splt.set_title("Message signal frequency domain")
    splt.set_xlabel("Frequency [Hz]")
    splt.set_ylabel("Power [W]")
    fig.canvas.set_window_title(str(i) + " Message signal")
    if savefig:
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(i) + "-message-signal.svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(i) + "-message-signal.png", bbox_inches='tight')
        print("Message signal saved.")

    i += 1
    df = output_frequencies[-1] / output_frequencies.size
    pos1_fd = int(carrier_frequency / df) - int(coarse_acquisition_code_frequency * 10 / df)
    pos2_fd = int(carrier_frequency / df) + int(coarse_acquisition_code_frequency * 10 / df)
    fig = plt.figure(i)
    splt = plt.subplot(211)
    plt.plot(t[pos1_td:pos2_td], output_signal[pos1_td:pos2_td], lw=0.3)
    splt.set_title("Output signal time domain")
    splt.set_xlabel("Time [s]")
    splt.set_ylabel("Amplitude [V]")
    splt = plt.subplot(212)
    plt.semilogy(output_frequencies[pos1_fd:pos2_fd:4], output_spectrum[pos1_fd:pos2_fd:4], lw=0.2)
    splt.set_title("Output signal frequency domain")
    splt.set_xlabel("Frequency [Hz]")
    splt.set_ylabel("Power [W]")
    fig.canvas.set_window_title(str(i) + " Output signal")
    if savefig:
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(i) + "-output-signal.svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(i) + "-output-signal.png", bbox_inches='tight')
        print("Output signal saved.")

    i += 1
    fig = plt.figure(i)
    splt = plt.subplot(211)
    plt.plot(t[pos1_td:pos2_td], input_signal[pos1_td:pos2_td], lw=0.3)
    splt.set_title("Input signal time domain")
    splt.set_xlabel("Time [s]")
    splt.set_ylabel("Amplitude [V]")
    splt = plt.subplot(212)
    plt.semilogy(input_frequencies[pos1_fd:pos2_fd:4], input_spectrum[pos1_fd:pos2_fd:4], lw=0.2)
    splt.set_title("Input signal frequency domain")
    splt.set_xlabel("Frequency [Hz]")
    splt.set_ylabel("Power [W]")
    fig.canvas.set_window_title(str(i) + " Input signal")
    if savefig:
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(i) + "-input-signal.svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(i) + "-input-signal.png", bbox_inches='tight')
        print("Input signal saved.")

    i += 1
    df = mixed_frequencies[-1] / mixed_frequencies.size
    fig = plt.figure(i)
    splt = plt.subplot(211)
    plt.plot(t[pos1_td:pos2_td], mixed_signal[pos1_td:pos2_td], lw=0.3)
    splt.set_title("Mixed signal time domain")
    splt.set_xlabel("Time [s]")
    splt.set_ylabel("Amplitude [V]")
    splt = plt.subplot(223)
    pos1_fd = 0
    pos2_fd = int(intermediate_frequency * 2 / df)
    plt.semilogy(mixed_frequencies[pos1_fd:pos2_fd:4], mixed_spectrum[pos1_fd:pos2_fd:4], lw=0.2)
    splt.set_title("Mixed signal frequency domain, low part")
    splt.set_xlabel("Frequency [Hz]")
    splt.set_ylabel("Power [W]")
    splt = plt.subplot(224)
    pos1_fd = int((intermediate_frequency + 2 * mixer_frequency) / df) - int(intermediate_frequency / df)
    pos2_fd = int((intermediate_frequency + 2 * mixer_frequency) / df) + int(intermediate_frequency / df)
    plt.semilogy(mixed_frequencies[pos1_fd:pos2_fd:4], mixed_spectrum[pos1_fd:pos2_fd:4], lw=0.2)
    splt.set_title("Mixed signal frequency domain, high part")
    splt.set_xlabel("Frequency [Hz]")
    splt.set_ylabel("Power [W]")
    fig.canvas.set_window_title(str(i) + " Mixed input signal")
    if savefig:
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(i) + "-mixed-input-signal.svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(i) + "-mixed-input-signal.png", bbox_inches='tight')
        print("Mixed input signal saved.")

    i += 1
    df = filtered_frequencies[-1] / filtered_frequencies.size
    pos1_fd = 0
    pos2_fd = int(sampling_frequency / df)
    fig = plt.figure(i)
    splt = plt.subplot(211)
    plt.plot(t[pos1_td:pos2_td], filtered_signal[pos1_td:pos2_td], lw=0.3)
    splt.set_title("Filtered signal time domain")
    splt.set_xlabel("Time [s]")
    splt.set_ylabel("Amplitude [V]")
    splt = plt.subplot(212)
    plt.semilogy(filtered_frequencies[pos1_fd:pos2_fd:4], filtered_spectrum[pos1_fd:pos2_fd:4], lw=0.2)
    splt.set_title("Filtered signal frequency domain")
    splt.set_xlabel("Frequency [Hz]")
    splt.set_ylabel("Power [W]")
    fig.canvas.set_window_title(str(i) + " Filtered input signal")
    if savefig:
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(i) + "-filtered-input-signal.svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(i) + "-filtered-input-signal.png", bbox_inches='tight')
        print("Filtered input signal saved.")

    i += 1
    pos1_td = int(2.4e-6 / sampled_dt) - int(0.7e-6 / sampled_dt)
    pos2_td = int(2.4e-6 / sampled_dt) + int(0.7e-6 / sampled_dt)
    df = sampled_frequencies[-1] / sampled_frequencies.size
    pos1_fd = 0
    pos2_fd = -1
    fig = plt.figure(i)
    splt = plt.subplot(211)
    plt.plot(sampled_t[pos1_td:pos2_td], sampled_signal[pos1_td:pos2_td], lw=0.3)
    splt.set_title("Sampled signal time domain")
    splt.set_xlabel("Time [s]")
    splt.set_ylabel("Amplitude [V]")
    splt = plt.subplot(212)
    plt.semilogy(sampled_frequencies[pos1_fd:pos2_fd:4], sampled_spectrum[pos1_fd:pos2_fd:4], lw=0.2)
    splt.set_title("Sampled signal frequency domain")
    splt.set_xlabel("Frequency [Hz]")
    splt.set_ylabel("Power [W]")
    fig.canvas.set_window_title(str(i) + " Sampled input signal")
    if savefig:
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(i) + "-sampled-input-signal.svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(i) + "-sampled-input-signal.png", bbox_inches='tight')
        print("Sampled input signal saved.")

    i += 1
    df = reference_carrier_frequencies[-1] / reference_carrier_frequencies.size
    pos1_fd = int(intermediate_frequency / df) - int(coarse_acquisition_code_frequency / df)
    pos2_fd = int(intermediate_frequency / df) + int(coarse_acquisition_code_frequency / df)
    fig = plt.figure(i)
    splt = plt.subplot(211)
    plt.plot(sampled_t[pos1_td:pos2_td], reference_carrier_signal[pos1_td:pos2_td], lw=0.3)
    splt.set_title("Reference carrier signal time domain")
    splt.set_xlabel("Time [s]")
    splt.set_ylabel("Amplitude [V]")
    splt = plt.subplot(212)
    plt.semilogy(reference_carrier_frequencies[pos1_fd:pos2_fd], reference_carrier_spectrum[pos1_fd:pos2_fd], lw=0.2)
    splt.set_title("Reference carrier signal frequency domain")
    splt.set_xlabel("Frequency [Hz]")
    splt.set_ylabel("Power [W]")
    if savefig:
        fig.canvas.set_window_title(str(i) + " Reference carrier signal")
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(i) + "-reference-carrier-signal.svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(i) + "-reference-carrier-signal.png", bbox_inches='tight')
        print("Reference carrier signal saved.")

    i += 1
    df = demodulated_frequencies[-1] / demodulated_frequencies.size
    pos1 = 0
    pos2 = int(digitized_coarse_acquisition_correlation_code.size * 1.7)
    fig = plt.figure(i)
    splt = plt.subplot(211)
    plt.plot(acquisition_signal_good[pos1:pos2], lw=0.3)
    splt.set_title(
        "Correlation of a good PRN code against the sampled input signal without the carrier with a 400 Hz Doppler shift")
    splt.set_xlabel("Offset")
    splt.set_ylabel("Match")
    splt = plt.subplot(212)
    plt.plot(acquisition_signal_bad[pos1:pos2], lw=0.3)
    splt.set_title(
        "Correlation of a bad PRN code against the sampled input signal without the carrier with a 400 Hz Doppler shift")
    splt.set_xlabel("Offset")
    splt.set_ylabel("Match")
    splt = plt.subplot(212)
    fig.canvas.set_window_title(str(i) + " Acquisition")
    if savefig:
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(i) + "-acquisition-doppler.svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(i) + "-acquisition-doppler.png", bbox_inches='tight')
        print("Acquisition saved.")

    i += 1
    df = demodulated_frequencies[-1] / demodulated_frequencies.size
    pos1_fd = 0
    pos2_fd = int(message_frequency * 4 / df)
    fig = plt.figure(i)
    splt = plt.subplot(221)
    plt.plot(sampled_t, demodulated_signal, lw=0.3)
    splt.set_title("Demodulated and digitized signal time domain")
    splt.set_xlabel("Time [s]")
    splt.set_ylabel("Amplitude [V]")
    splt = plt.subplot(222)
    plt.plot(sampled_t[::int(sampled_t.size / decoded_message_code.size)], decoded_message_code, lw=0.3)
    splt.set_title("Decoded message time domain")
    splt.set_xlabel("Time [s]")
    splt.set_ylabel("Bit value")
    splt = plt.subplot(212)
    plt.semilogy(demodulated_frequencies[pos1_fd:pos2_fd], demodulated_spectrum[pos1_fd:pos2_fd], lw=0.2)
    splt.set_title("Demodulated and digitized signal frequency domain")
    splt.set_xlabel("Frequency [Hz]")
    splt.set_ylabel("Power [W]")
    fig.canvas.set_window_title(str(i) + " Decoded message")
    if savefig:
        fig.set_size_inches((default_size[0] * 2, default_size[1] * 2))
        fig.savefig(str(i) + "-decoded-message.svg", bbox_inches='tight')
        fig.set_size_inches((default_size[0] * 3, default_size[1] * 3))
        fig.savefig(str(i) + "-decoded-message.png", bbox_inches='tight')
        print("Decoded message saved.")

    if showfig:
        plt.show()
