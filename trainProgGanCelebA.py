import pickle
import numpy as np
import tensorflow as tf
from models import ProgG, ProgD, global_variables, DTYPE

import matplotlib.pyplot as plt


# TRAINING PARAMETERS
NUM_SAMPLES = 81693
BATCH_SIZE = 32
NUM_STEPS_G_TRAIN = 3
STEP_VERBOSE = 100
SAMPLE_VERBOSE = 500
CHECKPOINT_PERIOD = 500
RES_LIST = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]
EPOCHS_PER_RES = {
    (4, 4): 2500,
    (8, 8): 2500,
    (16, 16): 2500,
    (32, 32): 2500,
    (64, 64): 2500
}
TRAIN_DATA_DIR = 'data/celeba/processed'
SAMPLE_IMGS_PATH = 'results/celeba'
MODEL_CHECKPOINT_PATH = 'checkpoints/celeba'


# MODELS PARAMETERS
GP_LAMBDA = 10
LATENT_DIM = 150
NUM_FINAL_CHANNELS = 3

# After THRESH_STAGE we want our fmaps to be equal to THRESH_FMAPS, but never greater than MAX_FMAPS
MAX_FMAPS = 512
THRESH_STAGE = 5
THRESH_FMAPS = 64
BASE_FMAPS = 8192 # 2**THRESH_STAGE * THRESH_FMAPS


# OPTIMIZERS PARAMETERS
D_OPT_LR = 0.0001
D_OPT_B1 = 0
D_OPT_B2 = 0.99

G_OPT_LR = 0.0001
G_OPT_B1 = 0
G_OPT_B2 = 0.99


def gradient_penalty(D_model, x_real, x_fake):
    # Pick random interpolation values
    lerp_eps = tf.constant(value=tf.random.uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0, maxval=1, dtype=DTYPE))

    # Interpolate generated and real images
    x_interp = lerp_eps * x_real + (1-lerp_eps) * x_fake

    # Calculate gradients w.r.t. input interpolated tensor for penalty
    with tf.GradientTape() as tape:
        tape.watch(x_interp)
        y_pred_interp = D_model.forward(x_interp)

    in_grads = tape.gradient(y_pred_interp, x_interp)

    # Compute the euclidean norm of gradients
    d_grad_norm = tf.square(in_grads)
    d_grad_norm = tf.reduce_sum(d_grad_norm, axis=np.arange(1, len(d_grad_norm.shape)))
    d_grad_norm = tf.sqrt(d_grad_norm)

    # Compute (1 - ||grad||)^2
    g_penalty = tf.square(1-d_grad_norm)
    return tf.reduce_mean(g_penalty)


def sample_images(epoch_idx: int, res: int, generator, rows: int = 5, cols: int = 5):
    z = tf.reshape(
        tf.random.normal(shape=[25 * LATENT_DIM], mean=0.0, stddev=1.0, dtype=DTYPE),
        shape=[25, LATENT_DIM]
    )

    fake_imgs = generator.forward(z)
    fake_imgs = fake_imgs.numpy()
    # Scale from a range [-1, 1] to a range [0, 1]
    fake_imgs = (fake_imgs + 1) / 2

    fig, axs = plt.subplots(rows, cols)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(fake_imgs[idx], cmap='binary')
            axs[i, j].axis('off')
            idx += 1

    fig.savefig(f"{SAMPLE_IMGS_PATH}/res{res}_epoch{epoch_idx}.png")
    plt.close()


def save_checkpoint(res: int, epoch: int):
    with open(f"{MODEL_CHECKPOINT_PATH}/param_dict_res{res}_ep{epoch}.pickle", "wb") as output_file:
        pickle.dump(global_variables, output_file)


def get_real_samples(res: tuple):
    h, w = res
    data_batch = []
    batch_idxs = np.random.randint(low=0, high=NUM_SAMPLES, size=BATCH_SIZE)
    for idx in batch_idxs:
        img = np.load(f'{TRAIN_DATA_DIR}/{h}x{w}/{idx}.npy')
        data_batch.append(img)

    return tf.constant(np.array(data_batch), dtype=DTYPE)


if __name__ == '__main__':
    # Init models
    g = ProgG(latent_dim=LATENT_DIM, num_channels=NUM_FINAL_CHANNELS, max_fmaps=MAX_FMAPS, base_fmaps=BASE_FMAPS)
    d = ProgD(num_channels=NUM_FINAL_CHANNELS, max_fmaps=MAX_FMAPS, base_fmaps=BASE_FMAPS)

    # Init optimizers
    d_optimizer = tf.optimizers.Adam(learning_rate=D_OPT_LR, beta_1=D_OPT_B1, beta_2=D_OPT_B2)
    g_optimizer = tf.optimizers.Adam(learning_rate=G_OPT_LR, beta_1=G_OPT_B1, beta_2=G_OPT_B2)

    # Loss history
    g_loss_hist = []
    d_loss_hist = []

    for res in RES_LIST:
        alpha_step = 1 / (EPOCHS_PER_RES[res] - EPOCHS_PER_RES[res] / 10)

        print(f"\nRES {res} -----------------------------------------")

        # Grow networks
        if res != (4, 4):
            g.grow()
            d.grow()

        # Reset lerp alpha
        curr_alpha = 0

        for epoch in range(EPOCHS_PER_RES[res]):

            z = tf.reshape(
                tf.random.normal(shape=[BATCH_SIZE*LATENT_DIM], mean=0.0, stddev=1.0, dtype=DTYPE),
                shape=[BATCH_SIZE, LATENT_DIM]
            )

            # Set models alpha values
            if curr_alpha < 1:
                curr_alpha += alpha_step
                curr_alpha = min(curr_alpha, 1.0)
                d.set_alpha(curr_alpha)
                g.set_alpha(curr_alpha)

            # Get batches of data
            real_imgs = get_real_samples(res)
            fake_imgs = g.forward(z)

            # Compute Wasserstein Loss GP for D
            with tf.GradientTape() as tape:
                d_params = global_variables['D']
                tape.watch(d_params)

                # Compute critic scores
                real_validity = d.forward(real_imgs)
                fake_validity = d.forward(fake_imgs)

                # Compute gradient penalty
                gp = gradient_penalty(d, real_imgs, fake_imgs)

                d_loss = -tf.reduce_mean(real_validity) + tf.reduce_mean(fake_validity) + GP_LAMBDA * gp

            # Compute Critics gradients
            d_grads = tape.gradient(d_loss, d_params)

            # Apply Critics gradients
            d_grads = tf.distribute.get_replica_context().all_reduce('sum', d_grads)
            # TODO: check if makes sense
            d_optimizer.apply_gradients(zip(d_grads, d_params), experimental_aggregate_gradients=False)
            # TODO: check if makes sense
            global_variables['D'] = d_params

            # Train Generator
            if epoch % NUM_STEPS_G_TRAIN == 0:
                z = tf.reshape(
                    tf.random.normal(shape=[BATCH_SIZE*LATENT_DIM], mean=0.0, stddev=1.0, dtype=DTYPE),
                    shape=[BATCH_SIZE, LATENT_DIM]
                )

                with tf.GradientTape() as tape:
                    g_params = global_variables['G']
                    tape.watch(g_params)

                    # Generate images
                    fake_imgs = g.forward(z)

                    # Predict generated images and compute Wasserstein Loss
                    fake_validity = d.forward(fake_imgs)
                    g_loss = -tf.reduce_mean(fake_validity)

                g_grads = tape.gradient(g_loss, g_params)
                g_grads = tf.distribute.get_replica_context().all_reduce('sum', g_grads)
                g_optimizer.apply_gradients(zip(g_grads, g_params), experimental_aggregate_gradients=False)
                global_variables['G'] = g_params

                # Append losses
                g_loss_hist.append(g_loss.numpy())
                d_loss_hist.append(d_loss.numpy())

            # Display losses
            if epoch % STEP_VERBOSE == 0:
                print(f'EPOCH: {epoch + 1}/{EPOCHS_PER_RES[res]}: g_loss: {g_loss}    |   d_loss: {d_loss}')

            # Sample images
            if epoch % SAMPLE_VERBOSE == 0:
                sample_images(epoch+1, res[0], g)

            # Save models params after
            if epoch % CHECKPOINT_PERIOD == 0:
                save_checkpoint(res=res[0], epoch=epoch)

        # Sample images and save params at the end of epoch
        sample_images(EPOCHS_PER_RES[res], res[0], g)
        save_checkpoint(res=res[0], epoch=EPOCHS_PER_RES[res])

    plt.figure()
    plt.style.use('seaborn')
    plt.title('Training loss')
    plt.plot(g_loss_hist, c='darkred', label='G_loss')
    plt.plot(d_loss_hist, c='c', label='D_loss')
    plt.legend()

    plt.xlabel('Checkpoint idx')
    plt.ylabel('Wasserstein GP loss')
    plt.show()
