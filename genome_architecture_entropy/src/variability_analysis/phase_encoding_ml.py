from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd


def select_region(data, chr, xmb, ymb, size, res_kb=50):
    seg_table_chr = data[data['chrom'] == chr]
    seg_table_chr = seg_table_chr.drop(columns=['chrom', 'start', 'stop'])
    seg_mat = np.array(seg_table_chr, dtype=bool)

    # base region to matrix coordinates
    x = int(xmb * 1000 / res_kb)
    y = int(ymb * 1000 / res_kb)
    s = int(size * 1000/ res_kb)

    # conversion to slices
    start = 0 + min(x, y)
    end = start + s
    seg_mat_region = seg_mat[start:end, :]
    
    return seg_mat_region

path = '/Users/pita/Documents/Rene/GAM_project/genome_architecture_entropy/data/experimental/'

data_paternal = path + 'F123.All.as.3NPs.mm10.curated.CAST.segregation_at50000.passed_qc_fc5_cw6_s11.table'
seg_table_paternal = pd.read_table(data_paternal)

data_maternal = path + 'F123.All.as.3NPs.mm10.curated.S129.segregation_at50000.passed_qc_fc5_cw6_s11.table'
seg_table_maternal = pd.read_table(data_maternal)

chrom = 'chr1'
x,y = 0,0
s = 10
print(chrom, x,y, s)

# Your original and convoluted datasets
seg_mat_paternal = select_region(seg_table_paternal, chrom, x, y, s)
seg_mat_maternal = select_region(seg_table_maternal, chrom, x, y, s)
seg_mat_combined = np.logical_or(seg_mat_paternal, seg_mat_maternal)

# Flatten your 2D datasets into 1D, because the Dense layer expects 1D inputs
original_data_paternal = seg_mat_paternal.flatten()
original_data_maternal = seg_mat_maternal.flatten()
convoluted_data = seg_mat_combined.flatten()

# Define the size of your datasets
original_size = original_data_paternal.size
convoluted_size = convoluted_data.size

# Define the encoding and decoding layers
encoding_layer = Dense(convoluted_size, activation='relu')
decoding_layer = Dense(original_size, activation='sigmoid')

# Define the encoder and decoder models
encoder_input = Input(shape=(original_size,))
encoder_output = encoding_layer(encoder_input)
encoder = Model(encoder_input, encoder_output)

decoder_input = Input(shape=(convoluted_size,))
decoder_output = decoding_layer(decoder_input)
decoder = Model(decoder_input, decoder_output)

# Define the autoencoder model
autoencoder_input = Input(shape=(original_size,))
autoencoder_output = decoder(encoder(autoencoder_input))
autoencoder = Model(autoencoder_input, autoencoder_output)

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder model
autoencoder.fit(original_data_paternal, original_data_paternal, epochs=50, batch_size=256, shuffle=True, validation_data=(convoluted_data, convoluted_data))

# Repeat the training for the maternal data
autoencoder.fit(original_data_maternal, original_data_maternal, epochs=50, batch_size=256, shuffle=True, validation_data=(convoluted_data, convoluted_data))

# Get the encoded datasets
encoded_original_data_paternal = encoder.predict(original_data_paternal)
encoded_original_data_maternal = encoder.predict(original_data_maternal)
encoded_convoluted_data = encoder.predict(convoluted_data)

# Get the decoded datasets
decoded_original_data_paternal = decoder.predict(encoded_original_data_paternal)
decoded_original_data_maternal = decoder.predict(encoded_original_data_maternal)
decoded_convoluted_data = decoder.predict(encoded_convoluted_data)

# Get the reconstructed datasets
reconstructed_original_data_paternal = autoencoder.predict(original_data_paternal)
reconstructed_original_data_maternal = autoencoder.predict(original_data_maternal)
reconstructed_convoluted_data = autoencoder.predict(convoluted_data)

# Get the loss of the reconstructed datasets
reconstructed_original_loss_paternal = np.mean(np.abs(reconstructed_original_data_paternal - original_data_paternal))
reconstructed_original_loss_maternal = np.mean(np.abs(reconstructed_original_data_maternal - original_data_maternal))
reconstructed_convoluted_loss = np.mean(np.abs(reconstructed_convoluted_data - convoluted_data))

# Get the loss of the decoded datasets
decoded_original_loss_paternal = np.mean(np.abs(decoded_original_data_paternal - original_data_paternal))
decoded_original_loss_maternal = np.mean(np.abs(decoded_original_data_maternal - original_data_maternal))
decoded_convoluted_loss = np.mean(np.abs(decoded_convoluted_data - convoluted_data))
