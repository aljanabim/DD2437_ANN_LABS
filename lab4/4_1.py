from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
plt.style.use("ggplot")

def plot_recon_error():
    '''
    Solves:
    Please investigate how the average reconstruction loss (the mean error
    between the original input and the reconstructed input) is aected by the
    number of hidden units decreasing from 500 down to 200.

    Result:
    Gives us a plot showing the reconstruction error on the traning data as a function of the number of hidden nodes 
    after completing a learning seqeunce of 20 epochs.


    Gives us a plot showing the average reconstruction error on the traning data as a function of the number of hidden nodes 
    after completing a learning seqeunce of 20 epochs, with the reconstruction error calculated every 5 epochs.
    '''
    PLOT_TYPE = 1
    
    n_hidden_min = 200
    n_hidden_max = 500
    
    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    try_range = range(n_hidden_min,n_hidden_max+1,100)
    avg_recon_error = np.zeros(len(try_range))
    all_recon_error = list(try_range)
    x_labels = list(try_range)
    epochs = 21
    i = 0
    for n_hidden in try_range:
        print("Testing n_hidden=", n_hidden)
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                    ndim_hidden=n_hidden,
                                    is_bottom=True,
                                    image_size=image_size,
                                    is_top=False,
                                    n_labels=10,
                                    batch_size=20)
        recon_loss, _ = rbm.cd1(visible_trainset=train_imgs, n_iterations=epochs)
        all_recon_error[i] = np.copy(recon_loss)
        avg_recon_error[i] = np.mean(recon_loss)
        i +=1
    if PLOT_TYPE == 1:
        plt.plot(x_labels,avg_recon_error, 'o')
        plt.xlabel("Number of hidden layers")
        plt.ylabel("Mean reconstruction error")
        plt.savefig('plots/4_1/recon_err_mean3.pdf')
        plt.show()
    if PLOT_TYPE == 2:
        print_period = 1 # Has to match the one in the RBM class
        x = list(range(0,epochs,print_period))
        for i in range(len(all_recon_error)):
            plt.plot(x,all_recon_error[i], label="Hidden layers:"+str(try_range[i]))
        plt.xlabel("Number of epochs")
        plt.ylabel("Reconstruction MSE")
        plt.legend()
        plt.savefig('plots/4_1/recon_err_tot3.pdf')
        plt.show()
        
def plot_weight_stability():
    '''
    More specically, please rst initialize the weight matrix (including hid-
    den and visible biases) with small random values (normally distributed,
    N(0,0.01)). Then, iterate the training process (CD) for the number of
    epochs varying between 10 and 20 for minibatches of size 20 (i.e. each
    epoch corresponds to a full swipe through a training set divided into mini-
    batches). The idea here is to obtain some level of convergence or stability
    in the behaviour of the units. How can you monitor and measure such
    stability?

    Result:
    Study stability interms of how the weights are updated after each epoch. We calculate the difference between every two consecutive weight updates and extract a stability 
    ratio defined as the number of weights that are the same in two consecutive updates divided by the total number of weights. As the network learns more about the data
    the stability ratio increases. I.e. the weight updates ratio decreses.

    Check if abs(weight_{i+1}-weight_{i})<0.001 within 0.001 in next step

    '''
    tries = 10
    res = np.array([])
    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    epochs = 21
    for i in range(tries):
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                    ndim_hidden=500,
                                    is_bottom=True,
                                    image_size=image_size,
                                    is_top=False,
                                    n_labels=10,
                                    batch_size=20)
        _, param_stability = rbm.cd1(visible_trainset=train_imgs, n_iterations=epochs)
        if i == 0:
            res = param_stability
        else:
            res = np.vstack((res,param_stability))
        # print(res)
    x = list(range(0,epochs-1))
    plt.plot(x,np.mean(res,axis=0))
    plt.xlabel("Number of epochs")
    plt.ylabel("Stability ratio")
    plt.savefig('plots/4_1/param_stability.pdf')
    plt.show()

if __name__ == "__main__":
    # plot_recon_error()
    # plot_weight_stability()

    # image_size = [28, 28]
    # train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
    #     dim=image_size, n_train=60000, n_test=10000)

    # ''' restricted boltzmann machine '''

    # print("\nStarting a Restricted Boltzmann Machine..")
    # print(test_lbls.shape)
    # rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
    #                                  ndim_hidden=500,
    #                                  is_bottom=True,
    #                                  image_size=image_size,
    #                                  is_top=False,
    #                                  n_labels=10,
    #                                  batch_size=10)

    # rbm.cd1(visible_trainset=train_imgs, n_iterations=20)
