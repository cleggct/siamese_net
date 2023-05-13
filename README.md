## siamese_net
An implementation of a siamese network model for unsupervised image feature representation.

# background
I needed an unsupervised model for image feature representation to use as part of another project I have been working on, so I put this little guy together in pytorch.
A siamese network is an unsupervised model that learns to distinguish between inputs. The way it does this is as follows: at each step of the training, the network
gets three images. The first image is passed through the network and its output is computed. The second image is chosen such that it is similar to the first image. In
my project, I did this by randomly zeroing some of the pixels in the first image. Then the network's output is computed on this similar image. Finally, the third image
is chosen such that it is different from the first (and second) image. The output of the network is computed on this image as well. Then the loss is computed as a
function of the similarity between the outputs on the first two images, and the difference between the outputs on the first and third image.

The tl;dr is that a siamese network learns to group similar inputs together and different inputs apart, and it does this by comparing outputs on similar pairs of images as well
as different ones.

# Methods
The network consisted of a fairly straight-forward CNN. I took some care with the dilation of each convolution in accordance with the method of dilated upsampling convolution. You
can read more about that here: https://arxiv.org/abs/1702.08502. The model was trained on the CIFAR-10 dataset, which consists of small (32 by 32) images each labeled as belonging
to one of ten categories. I thought working with smaller images would make things easier for my model, and since I was training this thing on my personal machine, it needed all the
help it could get! Images were normalized prior to being fed to the model. The output of the network is a 64-dimensional vector meant to give a feature representation of the input.

Initially I was having difficulty with the choice of loss function. MSE isn't very good because it tends to not be very sensitive to smaller differences in vectors, and additionally
the network was learning to minimize the MSE by essentially just outputting 0 for every input. Given these issues, you might consider another distance metric such as the L1 norm.
Using that as the loss function has the benefit of being sensitive enough to respond to even small differences in the vectors. However, the L1 norm would not solve the issue of
the network learning to simply output 0. Then I thought, perhaps normalizing the outputs could yield some results by forcing the vectors to have a minimum size, but this did not work
particularly well either. I was on the verge of sitting down and trying to scribble out a loss function myself when I heard about something called triplet loss, which was pretty much
designed for this exact situation. Here's a link to a paper about it: https://arxiv.org/abs/1412.6622. Apparently this thing's been around for a while.. I can't believe I never
heard about it before! Pytorch even has triplet loss as one of the built-in loss functions, so it was simply a matter of plugging that in and the rest was smooth sailing.

I used the t-SNE algorithm to visualize my results. As an aside, this was my first time learning about t-SNE and golly what a neat algorithm! Datapoints were colored according to their label.
This may seem odd given that the labels were not used to train the model and we would expect there to be a lot of variation between feature representations of images in the same category, 
simply due to the fact that pictures of objects in the same category are not necessarily going to be similar themselves. However, I needed some way to gauge whether images were being grouped 
by similarity so my hope was that we would nonetheless see some images with the same labels being grouped together in places.

# Results
Without further ado, here is the plot generated by t-SNE on the model outputs for each image.

![Figure_1](https://github.com/cleggct/siamese_net/assets/45923683/7598fa95-c80b-477f-a7ba-a653af6ea58a)

Wow, that's a little crowded! Judging by this plot, it would appear that objects of the same category are indeed being clustered with other objects from that category, though there is a lot
of variation of the outputs for images in the same category, as expected. Therefore I conclude that the model appears to be working as intended. Whoopee!