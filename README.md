# Beyond Back-Propagation

Trying to figure out a way to optimize neural-networks differently, that will possibly allow parallelization over the
layers
(inherently impossible in back-propagation).

# Related Work

## Representation Learning

### [A Simple Framework for Contrastive Learning of Visual Representations] (Feb 2020)

This paper presents SimCLR: a simple framework for contrastive learning of visual representations. \
The self-supervised task is to identify that different augmentations of the same image are the same.

![SimCLR architecture](images/simclr_architecture.png "SimCLR Architecture")

Take home messages:

- Composition of data augmentations is important.
- Adding a nonlinear transformation between the representation and the contrastive loss helps.
- Contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning.

### [Big Self-Supervised Models are Strong Semi-Supervised Learners] (Jun 2020)

Future read.

### [Representation Learning with Contrastive Predictive Coding] (Jul 2018)

Future read.

### [Context Encoders: Feature Learning by Inpainting] (Apr 2016)

Future read.

### [Predicting What You Already Know Helps: Provable Self-Supervised Learning] (Aug 2020)

Propose a mechanism based on conditional independence to formalize how solving certain pretext tasks can learn 
representations that provably decreases the sample complexity of downstream supervised tasks.

### [Putting An End to End-to-End: Gradient-Isolated Learning of Representations] (May 2019)

Train a neural-network in a self-supervised, local manner (i.e. without labels and without end-to-end backpropagation).

![Greedy InfoMax architecture](images/greedy_infomax_architecture.png "Greedy InfoMax architecture")

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

![Loco v.s. Greedy InfoMax](images/loco_vs_greedy_infomax.png "Loco v.s. Greedy InfoMax")

This simple design closes the performance gap between local learning and end-to-end contrastive learning algorithms for
the first time. Aside from standard ImageNet experiments, also show results on complex downstream tasks such as object
detection and instance segmentation.

Take home messages:

- The overlapping enables "communication" between lower and upper layers.
- Self-supervised local learning can reach the performance of supervised back-propagation learning.

## Synthetic Gradients

### [Decoupled Neural Interfaces using Synthetic Gradients] (Aug 2016)

Use auxiliary networks to decouple sub-graphs, enabling updating them independently and asynchronously.

![Decoupled Interfaces](images/decoupled_interfaces.png "Decoupled Interfaces")

In particular focus on using the modelled synthetic gradient in place of true back-propagated error gradients.

![Synthetic Gradients](images/synthetic_gradients.png "Synthetic Gradients")

Predicting the inputs to downstream layers is also possible, completely unlocking (i.e. forward-unlocking) the training.

![Completely Unlocked Model](images/completely_unlocked.png "Completely Unlocked Model")

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

![Hierarchical Digits Generation](images/hierarchical_digits_generation.png "Hierarchical Digits Generation")

### [Greedy Layerwise Learning Can Scale to ImageNet] (Dec 2018)

Show that greedy layerwise optimization can reach competitive performance on ImageNet.

![Greedy Layerwise Model](images/greedy_layerwise_model.png "Greedy Layerwise Model")

Take home messages:
- local optimization works.
- layerwise training increases linear separability of the different layers' activations.
- Using auxiliary network with more than 1 hidden layer works better.

### [Decoupled Greedy Learning of CNNs] (Jan 2019)

Show that the greedy layerwise model can be trained in parallel, 
with the possibility of adding a buffer between two adjacent layers to completely unlocking the training process.

![DGL v.s. DNI](images/dgl_vs_dni_models.png "DGL v.s. DNI")

### [Parallel Training of Deep Networks with Local Updates] (Dec 2020)

Provide the first large scale investigation into local update methods in both vision and language domains.

![Comparing Local Learning Methods](images/comparing_local_learning_methods.png "Comparing Local Learning Methods")

Take home messages:
- Same as LoCo (which was done in the self-supervised setting),
  overlapping of layers seems to help also in the supervised local learning framework.
- Gradients of earlier layers differ from the true gradients (of regular back-prop).
  ![Gradient Similarities](images/gradient_silimarity_local_and_global.png "Gradient Similarities")
- Global back-propagation demonstrates higher capacity, in that it is able to memorize the dataset
  better than local greedy backpropagation.
  ![Fitting to Random Labels](images/fitting_to_random_labels.png "Fitting to Random Labels")
- Local methods learn different features (e.g. the first conv filters "look" different).
  ![Filters Comparison](images/filters_comparison.png "Filters Comparison")

## Feedback Alignment

### [Random feedback weights support learning in deep neural networks] (Nov 2014)

Instead of multiplying the back-propagated gradients with the forward weight matrix transposed,
multiply by a random matrix. The motivation is to obtain a more biological plausible learning rule.

![Feedback Alignment](images/feedback_alignment.png "Feedback Alignment")

Take home messages:
- The network "learns to learn".
- The angle between the directions of feedback alignment and back-prop decreases to below 90 degrees,
  meaning the direction of progress is still "descending". \
  Interestingly, the angle between the pseudo-inverse of the forward weight matrix goes to zero.
  ![Angles Comparison](images/angles_comparison.png "Angles Comparison")

  Note that there is also the "Nature communications" version: [Random synaptic feedback weights support error backpropagation for deep learning]

### [Direct Feedback Alignment Provides Learning in Deep Neural Networks] (Sep 2016)

Instead of using a random matrix to multiply with the back-propagated gradient,
use a random matrix for multiplying the top error directly (without backward passing through the top layers).
This enables backward-unlocking of the training process.

![Direct Feedback Alignment](images/direct_feedback_alignment.png "Direct Feedback Alignment")

Take home messages:
- Learning is possible even when the feedback path is disconnected from the forward path.
- Performed experiments on small datasets such as MNIST and CIFAR. \
  Works okay, slightly worse performance than back-prop.

### [Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures] (Jun 2020)

Showed empirically that direct feedback alignment works on a variety of different and difficult tasks,
such as neural view synthesis (e.g. NeRF), click-through rate prediction with recommender systems,
geometric learning with graph-convolutional networks and NLP with transformers.

![Neural View Synthesis](images/nerf_comparison.png "Neural View Synthesis")

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

[Random feedback weights support learning in deep neural networks]: https://arxiv.org/pdf/1411.0247.pdf

[Random synaptic feedback weights support error backpropagation for deep learning]: https://www.nature.com/articles/ncomms13276.pdf

[Direct Feedback Alignment Provides Learning in Deep Neural Networks]: https://arxiv.org/pdf/1609.01596.pdf

[Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures]: https://arxiv.org/pdf/2006.12878.pdf

[Convex Optimization]: https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf

[Online Learning and Online Convex Optimization]: https://www.cs.huji.ac.il/~shais/papers/OLsurvey.pdf