self.n_samples = vis_trainset.shape[0]
            self.max_n_minibatches = self.n_samples/self.batch_size
            n_minibatches = int(self.n_samples/self.batch_size + 0.5)
            minibatch_folds = np.array((list(range(n_minibatches))*self.batch_size)[:self.n_samples])
            for it in range(n_iterations):
                np.random.shuffle(minibatch_folds)
                print(it)
                for fold_index in range(n_minibatches):
                    mini_batch = vis_trainset[minibatch_folds == fold_index]
                    mini_lbl_batch = lbl_trainset[minibatch_folds == fold_index]
                    # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.
                    hid_prob, hid_bin  = self.rbm_stack['vis--hid'].get_h_given_v_dir(mini_batch)
                    hid_pred, _  = self.rbm_stack['vis--hid'].get_v_given_h_dir(hid_bin)
                    self.rbm_stack['vis--hid'].update_generate_params(hid_bin,mini_batch,hid_pred)

                    pen_prob, pen_bin = self.rbm_stack['hid--pen'].get_h_given_v_dir(hid_bin) #hid_prob
                    pen_pred, _ = self.rbm_stack['hid--pen'].get_v_given_h_dir(pen_bin)
                    self.rbm_stack['hid--pen'].update_generate_params(pen_bin,hid_bin,pen_pred) # (=,hid_prob,=)

                    # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.
                    pen_bin_lbl = np.concatenate((pen_bin, mini_lbl_batch), axis=1)
                    top_h_prob, top_h_bin = self.rbm_stack['pen+lbl--top'].get_h_given_v(pen_bin_lbl)
                    top_v_prob, top_v_bin = self.rbm_stack['pen+lbl--top'].get_v_given_h(top_h_bin)
                    top_v_bin[:,-self.n_labels:] = mini_lbl_batch
                    
                    v_0 = np.copy(pen_bin_lbl)
                    h_0 = np.copy(top_h_bin)
                    for i in range(self.n_gibbs_wakesleep):
                        top_h_prob, top_h_bin = self.rbm_stack['pen+lbl--top'].get_h_given_v(top_v_bin)
                        top_v_prob, top_v_bin = self.rbm_stack['pen+lbl--top'].get_v_given_h(top_h_bin)
                        top_v_bin[:,-self.n_labels:] = mini_lbl_batch
                        # Try clamping the labels later if necessary
                    h_1 = np.copy(top_h_prob)
                    v_1 = np.copy(top_v_prob)
                    self.rbm_stack['pen+lbl--top'].update_params(v_0, h_0, v_1, h_1)
                    
                    # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.
                    down_hid_prob, down_hid_bin = self.rbm_stack['hid--pen'].get_v_given_h_dir(top_v_bin[:,:self.sizes['pen']])
                    down_hid_pred, _ = self.rbm_stack['hid--pen'].get_h_given_v_dir(down_hid_bin)
                    self.rbm_stack['hid--pen'].update_recognize_params(down_hid_bin, top_v_bin[:,:self.sizes['pen']] , down_hid_pred)

                    down_vis_prob, down_vis_bin = self.rbm_stack['vis--hid'].get_v_given_h_dir(down_hid_bin)
                    down_vis_pred, _ = self.rbm_stack['vis--hid'].get_h_given_v_dir(down_vis_bin)
                    self.rbm_stack['vis--hid'].update_recognize_params(down_vis_bin, down_hid_bin , down_vis_pred)
                                        
                    
                    
                    # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                    # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.
                    # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.
                    # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.
                    # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.
                    
                    # WAKE PHASE



        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """

        n_samples = true_img.shape[0]

        vis = true_img # visible layer gets the image data

        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels

        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top.
        # In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system).
        # In that case, divide into mini-batches.

        vis_activations = vis
        hid_probs, hid_activations = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_activations)
        pen_probs, pen_activations = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid_activations)
        pen_lbl_activations = np.concatenate((pen_activations, lbl), axis=1)

        for _ in range(self.n_gibbs_recog):
            top_probs, top_activations = self.rbm_stack["pen+lbl--top"].get_h_given_v(pen_lbl_activations)
            pen_lbl_probs, pen_lbl_activations = self.rbm_stack["pen+lbl--top"].get_v_given_h(top_activations)

        predicted_lbl = pen_lbl_activations[:, -self.sizes["lbl"]:]

        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))

        return
