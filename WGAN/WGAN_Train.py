from WGAN.Models.WGAN_Discriminator import Discriminator
from WGAN.Models.WGAN_Generator import Generator
import argparse
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from WGAN.Utils.Metrics import loss_fft,cross_correlation,dtw_distance
from WGAN.Utils.Plot_utils import plot_prediction_reference,plot_losses
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def loss_ones(logits):
	loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
												   labels=tf.ones_like(logits))
	return tf.reduce_mean(loss)

def loss_zeros(logits):
	loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
												   labels=tf.zeros_like(logits))
	return tf.reduce_mean(loss)

def gradient_penalty(discriminator,real_emg,fake_emg):
	batch_size = real_emg.shape[0]
	t = tf.random.uniform([batch_size,1,1])
	#[b,1,1] => [b,real_emg.shape[1],real_emg.shape[2]]
	t = tf.broadcast_to(t,real_emg.shape)
	real_emg = tf.cast(real_emg,dtype =tf.float32)

	interpolate = t * real_emg + (1.-t) * fake_emg

	with tf.GradientTape() as tape:
		tape.watch([interpolate])
		d_interpolate_logits = discriminator.model(interpolate)
	gradient = tape.gradient(d_interpolate_logits,interpolate)

	#gradient.shape : [b,w,c] => [b,-1]
	gradient = tf.reshape(gradient,[gradient.shape[0],-1])
	gp = tf.norm(gradient,axis=1) # Find the second norm [b]
	gp = tf.reduce_mean((gp-1.)**2) # Find mean square deviation
	return gp


def d_loss_function(generator,discriminator,noise,real_emg,trining):
	#1.  treat real_emg as real
	#2.  treat generated emg as fake
	fake_emg = generator.model(noise,trining)
	d_fake_logits = discriminator.model(fake_emg,trining)
	d_real_logits = discriminator.model(real_emg,trining)

	d_loss_real = loss_ones(d_real_logits)  # Calculate the loss of real data
	d_loss_fake = loss_zeros(d_fake_logits) # Calculate loss of fake data

	gp = gradient_penalty(discriminator,real_emg,fake_emg) # GP penalty item

	loss = d_loss_fake + d_loss_real + 10.* gp
	return loss
def g_loss_function(generator,discriminator,noise,trining):

	fake_emg = generator.model(noise,trining)
	d_fake_logits = discriminator.model(fake_emg,trining)
	loss = loss_ones(d_fake_logits)

	return loss

def save_sample(epoch,generator,discriminator,real_emg,noise,G = -1,ch=-1):
	# noise = real_emg[:,512:512+config['noise_dim'],:]
	# noise = tf.random.uniform([config['batch_size'], config['noise_dim']], minval=-1., maxval=1.)
	gen_emg = generator.model(noise,training=False)

	dis_emg = tf.nn.sigmoid(discriminator.model(gen_emg,training=False))

	noise = np.reshape(noise, (noise.shape[0], noise.shape[1]))
	gen_emg = np.reshape(gen_emg,(gen_emg.shape[0],gen_emg.shape[1]))
	dis_emg = np.reshape(dis_emg,(dis_emg.shape[0],dis_emg.shape[1]))

	np.savetxt('./Output/G'+str(G)+'/ch'+str(ch)+'/Noise_' + str(epoch) + '.csv', noise, delimiter=',')
	np.savetxt('./Output/G'+str(G)+'/ch'+str(ch)+'/Generated_' + str(epoch) + '.csv', gen_emg, delimiter=',')
	np.savetxt('./Output/G'+str(G)+'/ch'+str(ch)+'/Discriminated_' + str(epoch) + '.csv', dis_emg, delimiter=',')

	ref_emg = np.array(real_emg[:,:,:])

	ref_emg = np.reshape(ref_emg,(ref_emg.shape[0],ref_emg.shape[1]))
	np.savetxt('./Output/G'+str(G)+'/ch'+str(ch)+'/Reference_' + str(epoch) + '.csv', ref_emg, delimiter=',')

	plot_prediction_reference(gen_signal=gen_emg, ref_signal=ref_emg, epoch=epoch,G = G,ch=ch)
	return gen_emg
def train(config,G,ch):
	training = False
	#Set random seed
	tf.random.set_seed(22)
	np.random.seed(22)

	tired_emg = np.load('ActiveDatasetsWindowsPadded/tired/All_tired_windows_'+str(G)+'.npy')[:, :, ch:ch+1]
	emg =       np.load('ActiveDatasetsWindowsPadded/relax/All_relax_windows_'+str(G)+'.npy')[:, :, ch:ch+1]
	print('emg.shape:',emg.shape)

	generator = Generator(config,training = True)
	discriminator = Discriminator(config,training = True)
	if config['load_weights']:
		print('--------------------load_Discriminator_weights------------------')
		discriminator.model.load_weights('./SavedModels/Discriminator_100.h5')
		print('--------------------load_Generator_weights------------------')
		generator.model.load_weights('./SavedModels/Generator_100.h5')

	g_optimizer = Adam(learning_rate=config['generator_learning_rate'],beta_1=0.5)
	d_optimizer = Adam(learning_rate=config['discriminator_learning_rate'],beta_1=0.5)

	metrics = []  # Evaluation of generated signal by evaluation index
	for epoch in range(config['epochs']):

		idx = np.random.randint(0,emg.shape[0],config['batch_size'])
		real_emg = emg[idx]

		noise = tired_emg[idx]

		gen_emg = generator.model(noise)
		validated = tf.nn.sigmoid(discriminator.model(gen_emg))

		metrics_index = np.argmax(validated)

		generated = np.array(gen_emg[metrics_index]).flatten()
		reference = np.array(real_emg[metrics_index]).flatten()
		fft_metric, fft_ref,fft_gen = loss_fft(reference,generated)
		dtw_metric = dtw_distance(reference,generated)
		cc_metric = cross_correlation(reference,generated)

		#train discriminator
		with tf.GradientTape() as tape:
			d_loss = d_loss_function(generator,discriminator,noise,real_emg,training)
		gradient = tape.gradient(d_loss,discriminator.model.trainable_variables) # Calculate gradient
		d_optimizer.apply_gradients(zip(gradient,discriminator.model.trainable_variables))  # Optimizer update parameters

		#train generator
		with tf.GradientTape() as tape:
			g_loss = g_loss_function(generator,discriminator,noise,training)
		gradient = tape.gradient(g_loss,generator.model.trainable_variables)
		g_optimizer.apply_gradients(zip(gradient,generator.model.trainable_variables))

		print('Epoch %d [D loss: %f] [G loss: %f][FFT Metric: %f] [DTW Metric: %f] [CC Metric: %f][validated:%f]'%
			  (epoch,  float(d_loss), float(g_loss),   fft_metric,    dtw_metric,   cc_metric[0],   validated[0]))
		metrics.append([[float(d_loss)], [float(g_loss)], [fft_metric], [dtw_metric], [cc_metric[0]],[validated[0]]])
		# Save the sample, and set the number of times to save it every other time
		if epoch % config['sample_interval'] == 0:
			if config['save_sample']:  # Save generated samples
				save_sample(epoch,generator,discriminator,real_emg,noise,G,ch)
			if config['plot_losses']:  # Save various evaluation indicators
				plot_losses(metrics, epoch,G=G,ch=ch)
				np.savetxt('./Output/G'+str(G)+'/ch'+str(ch)+'/d_loss.txt', np.array(metrics)[:, 0], delimiter=',')
				np.savetxt('./Output/G'+str(G)+'/ch'+str(ch)+'/g_loss.txt', np.array(metrics)[:, 1], delimiter=',')
				np.savetxt('./Output/G'+str(G)+'/ch'+str(ch)+'/fft_metric.txt', np.array(metrics)[:, 2], delimiter=',')
				np.savetxt('./Output/G'+str(G)+'/ch'+str(ch)+'/dtw_metric.txt', np.array(metrics)[:, 3], delimiter=',')
				np.savetxt('./Output/G'+str(G)+'/ch'+str(ch)+'/cc_metric.txt', np.array(metrics)[:, 4], delimiter=',')
			if config['save_models']:  # Save model parameters
				discriminator.save(epoch,G=G,ch=ch)
				generator.save(epoch,G=G,ch=ch)
	# Save the latest variety after training
	save_sample(epoch,generator,discriminator, real_emg,noise,G,ch)
	discriminator.save(G=G,ch=ch)
	generator.save(G=G,ch=ch)
	plot_losses(metrics, epoch,G=G,ch=ch)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='EMG-GAN-Train')

	parser.add_argument('--config_json','-config',default='configuration.json',type=str)

	args = parser.parse_args()

	config_file = args.config_json
	with open(config_file) as json_file:
		config = json.load(json_file)
	train(config,G=6,ch=3)