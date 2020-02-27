from util import *
from sklearn.metrics import mean_squared_error
import sys
np.set_printoptions(threshold=sys.maxsize)

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end.
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """

        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size

        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size

        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0

        self.weight_v_to_h = None

        self.weight_h_to_v = None

        self.learning_rate = 0.01

        self.momentum = 0.7

        self.print_period = 5

        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 5, #5000, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }


        self.max_n_minibatches = 6000
        return


    def cd1(self,visible_trainset, n_iterations=10000):

        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        print ("learning CD1")
        visible_trainset = visible_trainset[:100]
        n_samples = visible_trainset.shape[0]
        n_minibatches = int(n_samples/self.batch_size + 0.5)
        minibatch_folds = np.array((list(range(n_minibatches))*self.batch_size)[:n_samples])
        weight_history = [None]*(n_iterations)
        self.recon_loss = np.zeros(int(np.ceil(n_iterations/self.print_period)))
        for it in range(n_iterations):
            # print("Iterations", it)
            np.random.shuffle(minibatch_folds)
            # print(np.min(self.weight_vh),np.max(self.weight_vh))
            weight_history[it] = np.copy(self.weight_vh)
            for fold_index in range(n_minibatches):
                if fold_index > self.max_n_minibatches:
                    break
                minibatch = visible_trainset[minibatch_folds == fold_index]

                v_activations_0 = minibatch # could also make it binary but on quick testing it seemed not a good idea.
                h_probs_0, h_activations_0 = self.get_h_given_v(v_activations_0)
                v_probs_1, _ = self.get_v_given_h(h_activations_0, sample=False)
                h_probs_1, h_activations_1 = self.get_h_given_v(v_probs_1)

                self.update_params(v_activations_0,h_activations_0,v_probs_1,h_probs_1)


            # Generate reconstructed images
            if it % self.rf["period"] == 0 and self.is_bottom:
                viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=it, grid=self.rf["grid"])


            # print progress
            if it % self.print_period == 0:
                self.get_reconstruction_loss(it, visible_trainset)
            # if it+1 == n_iterations:
            # self.get_reconstruction_loss(it, visible_trainset)

        param_stability = self.get_param_stability(weight_history)
        return self.recon_loss, param_stability

    def get_reconstruction_loss(self, iteration,visible_trainset):
        forward_pass, _ = self.get_h_given_v(visible_trainset)
        reconstruction, _ = self.get_v_given_h(forward_pass)
        loss = mean_squared_error(visible_trainset, reconstruction)
        self.recon_loss[iteration//self.print_period] = loss
        print ("iteration=%7d recon_loss=%4.4f"%(iteration, loss))

    def get_param_stability(self, weight_history):
        tol = 0.001
        n = len(weight_history)
        n_weights = weight_history[0].size
        ratio_correct = np.zeros(n-1)
        for i in range(n-1):
            ratio_correct[i] = np.count_nonzero(np.abs(weight_history[i]-weight_history[i+1])<tol)/n_weights
        return ratio_correct

    def update_params(self,v_0,h_0,v_k,h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters
        n_samples = v_0.shape[0]

        self.delta_bias_v = self.learning_rate*np.sum(v_0-v_k,axis=0)/n_samples
        self.delta_weight_vh = self.learning_rate*(np.dot(h_0.T,v_0).T-np.dot(h_k.T,v_k).T)/n_samples
        self.delta_bias_h = self.learning_rate*np.sum(h_0-h_k,axis=0)/n_samples

        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h

        return

    def get_h_given_v(self,visible_minibatch, sample=True):
        # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below)

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses undirected weight "weight_vh" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        n_samples = visible_minibatch.shape[0]

        output_shape = (n_samples,self.ndim_hidden)
        h_given_v_prob = sigmoid(self.bias_h + np.dot(visible_minibatch, self.weight_vh))
        if sample:
            h_given_v_activation = sample_binary(h_given_v_prob)
        else:
            h_given_v_activation = 0
        return h_given_v_prob, h_given_v_activation

    def get_v_given_h(self,hidden_minibatch, sample=True):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.2] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.

            return_shape = (n_samples, self.ndim_visible)

            v_inputs = self.bias_v + hidden_minibatch @ self.weight_vh.T
            v_inputs_pen = v_inputs[:, :-self.n_labels]
            v_inputs_lbl = v_inputs[:, -self.n_labels:]

            v_probs_pen = sigmoid(v_inputs_pen)
            v_probs_lbl = softmax(v_inputs_lbl)
            v_probs = np.concatenate((v_probs_pen, v_probs_lbl), axis=1)

            v_activations_pen = sample_binary(v_probs_pen)
            v_activations_lbl = sample_categorical(v_probs_lbl)
            v_activations = np.concatenate((v_activations_pen, v_activations_lbl), axis=1)


        else:
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)
            return_shape = (n_samples, self.ndim_visible)
            v_probs = sigmoid(self.bias_v + hidden_minibatch @ self.weight_vh.T)
            if sample:
                v_activations = (np.random.random(return_shape) < v_probs)
            else: v_activations=0

        return v_probs, v_activations


    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """


    def untwine_weights(self):

        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        output_shape = (n_samples,self.ndim_hidden)
        h_probs = sigmoid(self.bias_h + visible_minibatch @ self.weight_v_to_h)
        h_activations = sample_binary(h_probs)

        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below)

        return h_probs, h_activations

    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)

            raise Error("Should never be called when on top")

            pass

        else:

            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)
            output_shape = (n_samples, self.ndim_visible)
            v_probs = sigmoid(self.bias_v + hidden_minibatch @ self.weight_h_to_v)
            v_activations = (np.random.random(output_shape) < v_probs)

        return v_probs, v_activations

    def update_generate_params(self,inps,trgs,preds):

        """Update generative weight "weight_h_to_v" and bias "bias_v"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v

        return

    def update_recognize_params(self,inps,trgs,preds):

        """Update recognition weight "weight_v_to_h" and bias "bias_h"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h

        return
