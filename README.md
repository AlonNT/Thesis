# Table of Contents

- [Introduction](#introduction)
- [Experiments](#experiments)
  - [1st iteration](#1st-iteration)
  - [2nd iteration](#2nd-iteration)
- [Related Work](#related-work)
  - [Representation Learning](#representation-learning)
    - [SimCLR](#a-simple-framework-for-contrastive-learning-of-visual-representations-feb-2020)
    - [SimCLRv2](#big-self-supervised-models-are-strong-semi-supervised-learners-jun-2020)
    - [CPC](#representation-learning-with-contrastive-predictive-coding-jul-2018)
    - [Conext-Encoders](#context-encoders-feature-learning-by-inpainting-apr-2016)
    - [Predicting What You Already Know Helps](#predicting-what-you-already-know-helps-provable-self-supervised-learning-aug-2020)
    - [Putting An End to End-to-End](#putting-an-end-to-end-to-end-gradient-isolated-learning-of-representations-may-2019)
    - [LoCo](#loco-local-contrastive-representation-learning-aug-2020)
  - [Synthetic Gradients](#synthetic-gradients)
    - [Synthetic Gradients](#decoupled-neural-interfaces-using-synthetic-gradients-aug-2016)
    - [Understanding Synthetic Gradients](#understanding-synthetic-gradients-and-decoupled-neural-interfaces-mar-2017)
  - [Layerwise Optimization](#layerwise-optimization)
    - [A Provably Correct Algorithm for Deep Learning that Actually Works](#a-provably-correct-algorithm-for-deep-learning-that-actually-works-mar-2018)
    - [Greedy Layerwise Learning Can Scale to ImageNet](#greedy-layerwise-learning-can-scale-to-imagenet-dec-2018)
    - [Decoupled Greedy Learning of CNNs](#decoupled-greedy-learning-of-cnns-jan-2019)
    - [Parallel Training of Deep Networks with Local Updates](#parallel-training-of-deep-networks-with-local-updates-dec-2020)
    - [Training Neural Networks with Local Error Signals](#training-neural-networks-with-local-error-signals-jan-2019)
  - [Feedback Alignment](#feedback-alignment)
    - [Feedback Alignment](#random-feedback-weights-support-learning-in-deep-neural-networks-nov-2014)
    - [Direct Feedback Alignment](#direct-feedback-alignment-provides-learning-in-deep-neural-networks-sep-2016)
    - [Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures](#direct-feedback-alignment-scales-to-modern-deep-learning-tasks-and-architectures-jun-2020)
  - [Target Propagation](#target-propagation)
    - [Difference Target Propagation](#difference-target-propagation-dec-2014)
  - [Miscellaneous](#miscellaneous)
    - [Beyond Back-Propagation Survey](#training-deep-architectures-without-end-to-end-backpropagation-a-brief-survey-jan-2021)
  - [Books](#books)
    - [Convex Optimization](#convex-optimization)
    - [Online Learning and Online Convex Optimization](#online-learning-and-online-convex-optimization)

# Introduction

Trying to figure out a way to optimize neural-networks differently, that will possibly allow parallelization over the
layers
(inherently impossible in back-propagation).

# Experiments

## 1st iteration

I implemented Decoupled Greedy Learning (took some inspiration from [DGL repo])
I also use [DNI repo] which implemented "Decoupled Neural Interfaces" to incorporate synthetic gradients in my experiments.
I launched training on CIFAR-10 dataset.
The main architecture is (Conv-BatchNorm-ReLU-MaxPool) x 3, followed by a multi-layer-perceptron with 1 hidden layer.
Each conv layer has 128 channels with kernel-size 5 and resolution-preserving padding.
Each affine hidden layer in a multi-layer-perceptron contains 256 channels.
There are 4 variants to the model and training process:
- Regular: The main architecture without changes, as explained previously.
- DGL: Decoupled-Greedy-Learning \
  This model contains an auxiliary network after each block (i.e. Conv-BatchNorm-ReLU-MaxPool).
  Each auxiliary network is a multi-layer-perceptron with 1 hidden layer,
  which predicts the final target (i.e. the classes' scores),
  and it provides the gradient for the relevant block.
  Note that the last "auxiliary network" is actually a part of the final model. \
  The model is implemented in the class `CNNwDGL` in the file `model.py`.
- DNI: Decoupled-Neural-Interface \
  This model contains an auxiliary network after each block (i.e. Conv-BatchNorm-ReLU-MaxPool).
  Each auxiliary network is a CNN containing Conv-BatchNorm-Conv-BatchNorm-Conv
  which preserve the input tensor dimensions (both channels and spatial size).
  The auxiliary network predicts the back-propagated gradient of the downstream layers.
  The model is implemented in the class `CNNwDNI` in the file `model.py`.
- cDNI: Context-Decoupled-Neural-Interface.
  This is the same as DNI except that a "context" (i.e. the label) is provided for each auxiliary network.
  This is done by multiplying the one-hot vector representing the label with a linear layer
  and performs element-wise addition of the result with the output of the first conv layer.

The graphs describing the experiments results are attached.
Ignore the sudden "peaks", it is some sort of numerical issue with my accumulator of the loss and accuracy.

<p align="center">
<img src="images/train_loss_dgl_dni.svg" alt="Train loss" width="95%"/>
</p>
<p align="center">
<img src="images/test_loss_dgl_dni.svg" alt="Test loss" width="95%"/>
</p>
<p align="center">
<img src="images/train_acc_dgl_dni.svg" alt="Train Accuracy" width="95%"/>
</p>
<p align="center">
<img src="images/test_acc_dgl_dni.svg" alt="Test Accuracy" width="95%"/>
</p>

Conclusions:
- Both DGL and the regular model completely overfitted the training-data pretty fast,
  but DNI didn't, and cDNI "almost" did it but not completely. \
  This is probably due to the fact that synthetic gradients don't use the full capacity of the model which causes under-fitting.
- DGL slightly outperformed the regular model with respect to the test accuracy (77.6% v.s. 76.5%). \
  cDNI reached 73.7% (later slightly degraded to 71%-72%). \
  DNI lack behind with 65.6% (later degraded to 58%-60%). \
  My hypothesis is that DGL causes some sort of "regularization" which caused it to outperform
  the regular training, even though they both reached 0 loss.

## 2nd iteration

In the previous iteration, the (best) models reached about 76%-77%, which is below what is normally achieved on the CIFAR-10 dataset. \
In order to increase performance I added data-augmentations:
- Random horizontal flip.
- Random 32x32 crop (after padding with 4 to obtain 40x40 images).
- Data normalization helped a little bit (about 0.5%), 
  but there was no significant difference between normalizing to \[-1,+1\] or to a unit gaussian, 
  so for simplicity \[-1,+1\] was chosen for future experiments.
  
Adding data-augmentations significantly improved performance - from 76%-77% to 85%-86%.
Now there is even less difference between DGL and regular back-prop, and they both reached peak performance of 86.5% test accuracy. \
One notable difference is that DGL's train-loss in decreased slower than regular training, but eventually they both reached (almost) 0 loss.

<p align="center">
<img src="images/BasicCNN_DGL_vs_noDGL_train_loss.svg" alt="Train loss" width="95%"/>
</p>
<p align="center">
<img src="images/BasicCNN_DGL_vs_noDGL_test_loss.svg" alt="Test loss" width="95%"/>
</p>
<p align="center">
<img src="images/BasicCNN_DGL_vs_noDGL_train_acc.svg" alt="Train Accuracy" width="95%"/>
</p>
<p align="center">
<img src="images/BasicCNN_DGL_vs_noDGL_test_acc.svg" alt="Test Accuracy" width="95%"/>
</p>

Next, I implemented the VGG models (VGG11/13/16/19), in order to reproduce the performance in the literature which is above 90%. The model I use for the experiment is VGG16 which consists of the following modules: \
(Conv(64)-BatchNorm-ReLU) x 2 \
MaxPool \
(Conv(128)-BatchNorm-ReLU) x 2 \
MaxPool \
(Conv(256)-BatchNorm-ReLU) x 3 \
MaxPool \
(Conv(512)-BatchNorm-ReLU) x 3 \
MaxPool \
(Conv(512)-BatchNorm-ReLU) x 3 \
MaxPool \
Linear(10)
- Note that the VGG model was originally built for ImageNet dataset and not for CIFAR-10. \
  For example, it contains 5 down-sampling layers which cause the last convolutional block to reach spatial size of 1x1 whereas in ImageNet it's 7x7. \
  Additionally, the original VGG model contains additional 2 fully-connected layers with 4096 channels 
  which I omitted (as in the CIFAR-10 implementation of VGG I found online).

Comparing VGG16 to our previous BasicCNN shows increased performance (from 86.5% to 93.2%). DGL now performs worse than regular back-prop, achieving 89.5%.

<p align="center">
<img src="images/VGG16_DGL_vs_noDGL_train_loss.svg" alt="Train loss" width="95%"/>
</p>
<p align="center">
<img src="images/VGG16_DGL_vs_noDGL_test_loss.svg" alt="Test loss" width="95%"/>
</p>
<p align="center">
<img src="images/VGG16_DGL_vs_noDGL_train_acc.svg" alt="Train Accuracy" width="95%"/>
</p>
<p align="center">
<img src="images/VGG16_DGL_vs_noDGL_test_acc.svg" alt="Test Accuracy" width="95%"/>
</p>

Conclusions:
- DGL extends to other types of architectures like VGG, performing in the same ballpark as regular back-prop but slightly worse.
  -  It is possible that further hyper-parameters search might bridge the gap, as during my (informal) hyper-parameters search I noticed the increasing the learning-rate helped regular back-prop and harmed DGL training.
  - It is possible that different auxiliary networks will increase DGL performance. \
    Currently the auxiliary network used for DGL is one hidden layer MLP with 512 channels. 

# Related Work

## Representation Learning

### [A Simple Framework for Contrastive Learning of Visual Representations] (Feb 2020)

This paper presents SimCLR: A simple framework for contrastive learning of visual representations. \
The self-supervised task is to identify that different augmentations of the same image are the same.

<p align="center">
<img src="images/simclr_architecture.png" alt="SimCLR Architecture" width="70%"/>
</p>

Take home messages:

- Composition of data augmentations is important.
- Adding a nonlinear transformation between the representation and the contrastive loss helps.
- Contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning.

### [Big Self-Supervised Models are Strong Semi-Supervised Learners] (Jun 2020)

This paper presents SimCLRv2: A model based on SimCLR with improvements that reached new state-of-the-art. Generally, the training phase contains three stages: self-supervised pretraining, followed by supervised fine-tuning, and finally distillation with unlabeled examples. Additionally:
- The backbone which learns the representation in a self-supervised way is much larger than original SimCLR - ResNet-152 (3x+SK) v.s. ResNet-50 (4x).
- Later, in the third stage, the model can be made smaller via student-teacher distillation.
- The model uses several "deep-learning tricks" that "make better use of the parameters", such as selective kernels, channel-wise attention mechanism, etc.

<p align="center">
<img src="images/SimCLRv2_general_idea.png" alt="SimCLRv2 framework" width="60%"/>
</p>

Take home messages:
- Notions of task-agnostic v.s. task-specific use of unlabeled data. 
  The first stage (self-supervised pretraining as in original SimCLR) is task-agnostic, 
  whereas the last stage of distillation via unlabeled examples is task-specific.
- The fewer the labels, the more it is possible to benefit from a bigger model.
  <p align="center">
  <img src="images/SimCLRv2_improvements_per_size.png" alt="Improvement per #parameters" width="40%"/>
  </p>
- Bigger / deeper projection heads improve representation learning.
  <p align="center">
  <img src="images/SimCLRv1_head_size.png" alt="Bigger heads" width="40%"/>
  </p>

### [Representation Learning with Contrastive Predictive Coding] (Jul 2018)

Propose a universal unsupervised learning approach to extract useful representations called Contrastive Predictive Coding. Use a probabilistic contrastive
loss which induces the latent space to capture information that is maximally useful to predict future samples. Demonstrate the approach on speech, images, text and RL.

<p align="center">
<img src="images/cpc_arch.png" alt="CPC architecture" width="70%"/>
</p>

On images (which are not time-series) the model split an image to a grid of overlapping patches, and uses a PixelCNN-style autoregressive model to make predictions about the latent activations in following rows top-to-bottom.

<p align="center">
<img src="images/cpc_for_images.png" alt="CPC for images" width="60%"/>
</p>

Take home messages:
- They show theoretically that optimizing the InfoNCE loss maximizes a lower bound on the mutual information between the future data-point representation and the context vector (summarizing the past).
- In its time, reached state-of-the-art on unsupervised classification, but this was still quite far from supervised learning.

### [Context Encoders: Feature Learning by Inpainting] (Apr 2016)

Present Context-Encoders - a convolutional neural network trained to generate the contents of an arbitrary image region conditioned on its surroundings. In order to succeed at this task, context encoders need to both understand the content of the entire image. Quantitatively demonstrate the effectiveness of the features for CNN pre-training on classification, detection, and segmentation tasks.

<p align="center">
<img src="images/inpainting_example.png" alt="Inpainting example" width="40%"/>
</p>

<p align="center">
<img src="images/context_encoder_arch.png" alt="Context Encoder Architecture" width="90%"/>
</p>

Take home messages:
- Similar in spirit to auto-encoders and denoising auto-encoders. \
  However, in auto-encoders the representation is likely to just compress the image content without learning a semantically meaningful representation. \ 
  Denoising auto-encoders is more similar in spirit, but here a large region is missing and not just localized and low-level corruption, 
  so the high level semantics of the image need to be understood.

### [Predicting What You Already Know Helps: Provable Self-Supervised Learning] (Aug 2020)

Propose a mechanism based on conditional independence to formalize how solving certain pretext tasks can learn 
representations that provably decreases the sample complexity of downstream supervised tasks.

### [Putting An End to End-to-End: Gradient-Isolated Learning of Representations] (May 2019)

Train a neural-network in a self-supervised, local manner (i.e. without labels and without end-to-end backpropagation).

<p align="center">
<img src="images/greedy_infomax_architecture.png" alt="Greedy InfoMax architecture" width="60%"/>
</p>

It uses the InfoNCE objective developed in CPC paper ([Representation Learning with Contrastive Predictive Coding]). \
Essentially, this objective pairs the representations of temporally nearby patches and contrasts them against random
pairs. Therefore, as shown in CPC paper, it maximizes the mutual information between temporally nearby representations.

Take home messages:

- Local learning is possible in the regime of self-supervised learning.
- Interesting self-supervised task - maximize the mutual information between temporally nearby representations
  (e.g. different patches of the same image).

### [LoCo: Local Contrastive Representation Learning] (Aug 2020)

Show that by overlapping local blocks stacking on top of each other, we effectively increase the decoder depth and allow
upper blocks to implicitly send feedbacks to lower blocks.

<p align="center">
<img src="images/loco_vs_greedy_infomax.png" alt="Loco v.s. Greedy InfoMax" width="90%"/>
</p>

This simple design closes the performance gap between local learning and end-to-end contrastive learning algorithms for
the first time. Aside from standard ImageNet experiments, also show results on complex downstream tasks such as object
detection and instance segmentation.

Take home messages:

- The overlapping enables "communication" between lower and upper layers.
- Self-supervised local learning can reach the performance of supervised back-propagation learning.

## Synthetic Gradients

### [Decoupled Neural Interfaces using Synthetic Gradients] (Aug 2016)

Use auxiliary networks to decouple sub-graphs, enabling updating them independently and asynchronously.

<p align="center">
<img src="images/decoupled_interfaces.png" alt="Decoupled Interfaces" width="40%"/>
</p>

In particular focus on using the modelled synthetic gradient in place of true back-propagated error gradients.

<p align="center">
<img src="images/synthetic_gradients.png" alt="Synthetic Gradients" width="20%"/>
</p>

Predicting the inputs to downstream layers is also possible, completely unlocking (i.e. forward-unlocking) the training.

<p align="center">
<img src="images/completely_unlocked.png" alt="Completely Unlocked Model" width="70%"/>
</p>

Take home messages:

- Notions of backward-locking, update-locking and forward-locking. \
  All of them are possible (to some extent).
- It works, but (quite) worse than regular back-propagation.

### [Understanding Synthetic Gradients and Decoupled Neural Interfaces] (Mar 2017)

Provide some theoretical explanations to synthetic gradients, for example:

- The critical points of the original optimization problem are maintained when using synthetic-gradients.
- Analyze the learning dynamics of synthetic gradients.

## Layerwise Optimization

### [A Provably Correct Algorithm for Deep Learning that Actually Works] (Mar 2018)

Create a toy dataset containing digits that are generated hierarchically, and prove layerwise optimization works.

<p align="center">
<img src="images/hierarchical_digits_generation.png" alt="Hierarchical Digits Generation" width="90%"/>
</p>

### [Greedy Layerwise Learning Can Scale to ImageNet] (Dec 2018)

Show that greedy layerwise optimization can reach competitive performance on ImageNet.

<p align="center">
<img src="images/greedy_layerwise_model.png" alt="Greedy Layerwise Model" width="90%"/>
</p>

Take home messages:
- local optimization works.
- layerwise training increases linear separability of the different layers' activations.
- Using auxiliary network with more than 1 hidden layer works better.

### [Decoupled Greedy Learning of CNNs] (Jan 2019)

Show that the greedy layerwise model can be trained in parallel, 
with the possibility of adding a buffer between two adjacent layers to completely unlocking the training process.

<p align="center">
<img src="images/dgl_vs_dni_models.png" alt="DGL v.s. DNI" width="90%"/>
</p>

### [Parallel Training of Deep Networks with Local Updates] (Dec 2020)

Provide the first large scale investigation into local update methods in both vision and language domains.

<p align="center">
<img src="images/comparing_local_learning_methods.png" alt="Comparing Local Learning Methods" width="90%"/>
</p>

Take home messages:
- Same as LoCo (which was done in the self-supervised setting),
  overlapping of layers seems to help also in the supervised local learning framework.
- Gradients of earlier layers differ from the true gradients (of regular back-prop).
  <p align="center">
  <img src="images/gradient_silimarity_local_and_global.png" alt="Gradient Similarities" width="30%"/>
  </p>
- Global back-propagation demonstrates higher capacity, in that it is able to memorize the dataset
  better than local greedy backpropagation.
  <p align="center">
  <img src="images/fitting_to_random_labels.png" alt="Fitting to Random Labels" width="30%"/>
  </p>
- Local methods learn different features (e.g. the first conv filters "look" different).
  <p align="center">
  <img src="images/filters_comparison.png" alt="Filters Comparison" width="80%"/>
  </p>

### [Training Neural Networks with Local Error Signals] (Jan 2019)

Use single-layer auxiliary-networks and two different supervised loss functions to generate local error signals, 
and show that the combination of these losses helps.

<p align="center">
<img src="images/predsim_arch.png" alt="DGL v.s. DNI" width="70%"/>
</p>

In addition to predicting the target classes' scores, they add another a 'similiarity loss' which encourages distinct classes to have distinct representations.
They perform experiments using VGG-like networks on a lot of datasets, approaching and sometime surpassing the performance of regular back-prop. For example, in CIFAR-10 reached 95%-96% test accuracy.

Take home messages:
- Similiary loss works quite good, and somehow complementary to regular prediction loss (indeed, using combination of these losses helps).

## Target Propagation

### [Difference Target Propagation] (Dec 2014)

Associate with each feedforward unit’s activation value a **target value** rather than a **loss gradient**. 
The target value is meant to be close to the activation value while being likely to have provided a smaller loss 
(if that value had been obtained in the feedforward phase). 
Suppose the network structure is defined by 
<p align="center">
<img src="images/tp_formula_1.png" alt="tp_formula_1" width="40%"/>
</p>
where h_i is the state of the i-th hidden layer (h_M is the output and h_0 is the input). 
Define 
<p align="center">
<img src="images/tp_formula_2.png" alt="tp_formula_2" width="5%"/>
</p>
The set of parameters defining the mapping between the i-th and the j-th layer. 
We can now write h_j as a function of h_i by
<p align="center">
<img src="images/tp_formula_3.png" alt="tp_formula_3" width="20%"/>
</p>
The loss depends on the state of the i-th layer as follows:
<p align="center">
<img src="images/tp_formula_4.png" alt="tp_formula_4" width="40%"/>
</p>
The basic idea of target-propagation is to assign to each h_i a nearby value \hat{h_i} which (hopefully) leads to a lower global loss:
<p align="center">
<img src="images/tp_formula_5.png" alt="tp_formula_5" width="40%"/>
</p>
Such a \hat{h_i} is called a **target** for the i-th layer. Now update the weights of the i-th layer to minimize the MSE loss
<p align="center">
<img src="images/tp_formula_6.png" alt="tp_formula_6" width="40%"/>
</p>

Now, the top layer target should be directly driven from the gradient of the global loss (e.g. \hat{h_M} = y). 
For earlier layers take advantage of an "approximate inverse". Suppose for each f_i we have a function g_i such that 
<p align="center">
<img src="images/tp_formula_7.png" alt="tp_formula_7" width="40%"/>
</p>
Then choosing 
<p align="center">
<img src="images/tp_formula_8.png" alt="tp_formula_8" width="30%"/>
</p>
would have the consequence that (under some smoothness assumptions on f and g) minimizing the distance between h_{i−1} and \hat{h_{i-1}} should also minimize the loss L_i of the i-th layer. 
Obtaining such an inverse function is done using auto-encoder. 
This idea is illustrated here
<p align="center">
<img src="images/tp_figure.png" alt="tp_figure" width="40%"/>
</p>

This paper shows that a linear correction for the imperfectness of the auto-encoders, called **difference** target propagation, is very effective to make target propagation actually work. It basically defines a different target 
<p align="center">
<img src="images/tp_formula_9.png" alt="tp_formula_9" width="40%"/>
</p>
which helps (read the paper for full details).

Take home messages:
- Interesting idea. 
- Unfortunately, seems to work worse than the alternatives (proxy objectives and synthetic gradients). \
  Furthermore, experiments were done using only fully-connected networks (MNIST and CIFAR-10).

## Miscellaneous

### [Training Deep Architectures Without End-to-End Backpropagation: A Brief Survey] (Jan 2021)

A nice survey of alternatives to back-propagation. It covers three main topics:
- Proxy Objectives - Use some local objective for each layer. \
  Covers:
  - Two papers of the survey authors - encourage representations of same-class samples to be closer and different-class samples to be distinct.
  - Belilovsky et al. (2019;2020).
  - Nøkland et al. (2019) - basically a combination of Belilovsky et al. and the survey authors' papers.
  - Two papers which resemble DGL but differ in their auxiliary networks - one uses a fixed random weighted fully-connected layer, and the other uses trainable two layered fully-connected network.
- Target Propagation - Use specific target tensor for each layer. \
  Covers:
  - "Difference target propagation" and "How auto-encoders could provide credit assignment in deep networks via target propagation".
  - "A theoretical framework for target propagation".
- Synthetic Gradients \
  Covers the original paper and the following paper giving some theoretical insights.

<p align="center">
<img src="images/beyond_back_prop_survey.png" alt="Beyond Back-Propagation Survey" width="90%"/>
</p>

Take home messages:
- Trying to make the representations of same-class samples to be close and different-class samples to be far 
  seems like an interesting and somewhat "popular" idea. 
  Both the authors and Nøkland et al. (2019) found it works, but there are no good explanations. \
  It requires "less labels" - only same/not-same class, and not the actual class.
- The authors papers seem to have interesting theoretical claims - worth a read.
- Target propagation seems interesting and worth a read. However it seems to work worse than proxy objectives. \
  The new paper from 2020 shows "Direct TP" which is similar in spirit to "Direct FA".

## Feedback Alignment

### [Random feedback weights support learning in deep neural networks] (Nov 2014)

Instead of multiplying the back-propagated gradients with the forward weight matrix transposed,
multiply by a random matrix. The motivation is to obtain a more biological plausible learning rule.

<p align="center">
<img src="images/feedback_alignment.png" alt="Feedback Alignment" width="30%"/>
</p>

Take home messages:
- The network "learns to learn".
- The angle between the directions of feedback alignment and back-prop decreases to below 90 degrees,
  meaning the direction of progress is still "descending". \
  Interestingly, the angle between the pseudo-inverse of the forward weight matrix goes to zero.
  
  <p align="center">
  <img src="images/angles_comparison.png" alt="Angles Comparison" width="40%"/>
  </p>

  Note that there is also the "Nature communications" version: [Random synaptic feedback weights support error backpropagation for deep learning]

### [Direct Feedback Alignment Provides Learning in Deep Neural Networks] (Sep 2016)

Instead of using a random matrix to multiply with the back-propagated gradient,
use a random matrix for multiplying the top error directly (without backward passing through the top layers).
This enables backward-unlocking of the training process.

<p align="center">
<img src="images/direct_feedback_alignment.png" alt="Direct Feedback Alignment" width="50%"/>
</p>

Take home messages:
- Learning is possible even when the feedback path is disconnected from the forward path.
- Performed experiments on small datasets such as MNIST and CIFAR. \
  Works okay, slightly worse performance than back-prop.

### [Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures] (Jun 2020)

Showed empirically that direct feedback alignment works on a variety of different and difficult tasks,
such as neural view synthesis (e.g. NeRF), click-through rate prediction with recommender systems,
geometric learning with graph-convolutional networks and NLP with transformers.

<p align="center">
<img src="images/nerf_comparison.png" alt="Neural View Synthesis" width="90%"/>
</p>

Take home messages:
- Seems to work okay but not as good as back-prop.

## Books

### [Convex Optimization]

### [Online Learning and Online Convex Optimization]

[A Simple Framework for Contrastive Learning of Visual Representations]: https://arxiv.org/pdf/2002.05709.pdf

[Big Self-Supervised Models are Strong Semi-Supervised Learners]: https://arxiv.org/pdf/2006.10029.pdf

[Representation Learning with Contrastive Predictive Coding]: https://arxiv.org/pdf/1807.03748.pdf

[Context Encoders: Feature Learning by Inpainting]: https://arxiv.org/pdf/1604.07379.pdf

[Predicting What You Already Know Helps: Provable Self-Supervised Learning]: https://arxiv.org/pdf/2008.01064.pdf

[Putting An End to End-to-End: Gradient-Isolated Learning of Representations]: https://arxiv.org/pdf/1905.11786.pdf

[LoCo: Local Contrastive Representation Learning]: https://arxiv.org/pdf/2008.01342.pdf

[Decoupled Neural Interfaces using Synthetic Gradients]: https://arxiv.org/pdf/1608.05343.pdf

[Understanding Synthetic Gradients and Decoupled Neural Interfaces]: https://arxiv.org/pdf/1703.00522.pdf

[A Provably Correct Algorithm for Deep Learning that Actually Works]: https://arxiv.org/pdf/1803.09522.pdf

[Greedy Layerwise Learning Can Scale to ImageNet]: https://arxiv.org/pdf/1812.11446.pdf

[Decoupled Greedy Learning of CNNs]: https://arxiv.org/pdf/1901.08164.pdf

[Parallel Training of Deep Networks with Local Updates]: https://arxiv.org/pdf/2012.03837.pdf

[Training Neural Networks with Local Error Signals]: https://arxiv.org/pdf/1901.06656.pdf

[Difference Target Propagation]: https://arxiv.org/pdf/1412.7525.pdf

[Training Deep Architectures Without End-to-End Backpropagation: A Brief Survey]: https://arxiv.org/pdf/2101.03419.pdf

[Random feedback weights support learning in deep neural networks]: https://arxiv.org/pdf/1411.0247.pdf

[Random synaptic feedback weights support error backpropagation for deep learning]: https://www.nature.com/articles/ncomms13276.pdf

[Direct Feedback Alignment Provides Learning in Deep Neural Networks]: https://arxiv.org/pdf/1609.01596.pdf

[Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures]: https://arxiv.org/pdf/2006.12878.pdf

[Convex Optimization]: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf

[Online Learning and Online Convex Optimization]: https://www.cs.huji.ac.il/~shais/papers/OLsurvey.pdf

[DGL repo]: https://github.com/eugenium/DGL

[DNI repo]: https://github.com/koz4k/dni-pytorch
