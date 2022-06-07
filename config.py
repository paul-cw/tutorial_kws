## depend on machine
data_dir = './data/' ## where google speech command set is downloaded to or already resides, relative works, absolute should work as well
output_dir = './' ## where we write training outputs like confusion matrices to. relative works, absolute should work as well


## global, better keep them as is
fs = 16000  # sample frequency
l= 40  # frame length in ms
s= 20  # stride in ms 
n_mfccs= 40  # number of mfcc coeccificients

brn_directory= data_dir + '_background_noise_' # this is fixed for the download link we provide

kw     = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown']  # default adds unknown
kws_all= ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'unknown', 'silence']  
