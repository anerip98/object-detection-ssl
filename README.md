# Course Project for Deep Learning - Spring 2022 at NYU

## Contributors 
[Aneri Patel](https://github.com/anerip98)
[Mohammed Khalfan](https://github.com/mohammedkhalfan)
[Xiang Pan](https://github.com/Xiang-Pan)

This is the code repository for object detection using self supervised methods - VICReg and FasterRCNN/DETR. This work was done for the Deep Learning Spring'22 course at NYU.

The dataset is an unknown dataset provided for the course project. The dataset contains:
* 512,000 unlabeled images
* 30,000 labeled train images
* 20,000 labeled validation images

The submitted model was later evaluated on another unknown dataset.

Here, we pre-train a ResNet-50 backbone using VICReg and later finetune the model using FasterRCNN for object detection. We try finetuning using DETR and find that FasterRCNN converges faster. We use the hyperparameters mentioned in the reference papers and train on 4 V100 GPUs on the NYU HPC.

References:

[1] Bardes, Adrien, Jean Ponce, and Yann LeCun. "Vicreg: Variance-invariance-covariance regularization for self-supervised learning." arXiv preprint arXiv:2105.04906 (2021).

[2] Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." Advances in neural information processing systems 28 (2015).

[3] Zhu, Xizhou, et al. "Deformable detr: Deformable transformers for end-to-end object detection." arXiv preprint arXiv:2010.04159 (2020).
