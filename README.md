#  Generative adversarial network 
 
 The aim of this project was to train two neural networks using the MNIST digit data. 
 The way Generative Adversarial Networks are setup is that one network (a.k.a Generator) tries to create fake data by learning the data from the original dataset and then try to fool the other network (a.k.a Discriminator). The Discriminator, unaware of the data origin, tries to differentiate whether the data provided was from the original dataset or from the Generator. After the Discriminator predicts the output, the losses are then used by the Generator to update the weights and generate data more similar to the original one.
 
 The network was trained for 200 epochs and trained models can be found inside the <b>static</b> folder.
 
 Some images from the original dataset are:
 <br/><br/>
 <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/original_images/orig_0.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/original_images/orig_1.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/original_images/orig_2.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/original_images/orig_3.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/original_images/orig_4.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/original_images/orig_5.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/original_images/orig_6.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/original_images/orig_7.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/original_images/orig_8.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/original_images/orig_9.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/original_images/orig_10.png">
 
 And some of the generated images are:
 <br/>
<img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/generated_sample/generated_0.png"><img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/generated_sample/generated_1.png"><img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/generated_sample/generated_2.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/generated_sample/generated_3.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/generated_sample/generated_4.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/generated_sample/generated_5.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/generated_sample/generated_6.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/generated_sample/generated_7.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/generated_sample/generated_8.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/generated_sample/generated_9.png"> <img src="https://github.com/saini-vishal/-Generative-adversarial-network-/blob/master/app/generated_sample/generated_10.png">
 
 <br/>
 As it can be seen that the results are nearly identical and not significantly differentiable with the human eye.
 With a few more epochs training the results can be improved.
 
 The webapp for the following project is deployed on <a href="http://18.217.205.147:5000/">Amazon EC2</a>.
 Check it out!
