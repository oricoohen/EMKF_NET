v_db   = 0   # in dB, = 10*log10(r2/q2)
snr_db = -10   # in dB, paper convention

# compute variances:
r2 = 10.0 ** (-snr_db / 10.0)
q2 = r2 / (10.0 ** (v_db / 10.0))
print('r2_10=',r2)
print('q2_10=',q2)


snr_db = 1
# compute variances:
r2 = 10.0 ** (-snr_db / 10.0)
q2 = r2 / (10.0 ** (v_db / 10.0))
print('r2_1=',r2)
print('q2_1=',q2)


snr_db = 10
# compute variances:
r2 = 10.0 ** (-snr_db / 10.0)
q2 = r2 / (10.0 ** (v_db / 10.0))
print('r2_10=',r2)
print('q2_10=',q2)


snr_db = 20
# compute variances:
r2 = 10.0 ** (-snr_db / 10.0)
q2 = r2 / (10.0 ** (v_db / 10.0))
print('r2_20=',r2)
print('q2_20=',q2)



snr_db = 30
# compute variances:
r2 = 10.0 ** (-snr_db / 10.0)
q2 = r2 / (10.0 ** (v_db / 10.0))
print('r2_30=',r2)
print('q2_30=',q2)