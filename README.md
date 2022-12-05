# Adversarial-Training-Vision-Transformers

Vision Transformer on CIFAR-10

Vision Transformer (ViT) by Google Brain
The Vision Transformer (ViT) is basically BERT, but applied to images. It attains excellent results compared to state-of-the-art convolutional networks. In order to provide images to the model, each image is split into a sequence of fixed-size patches (typically of resolution 16x16 or 32x32), which are linearly embedded. One also adds a [CLS] token at the beginning of the sequence in order to classify images. Next, one adds absolute position embeddings and provides this sequence to the Transformer encoder.

# Use the notebook **Vision Transformer on CIFAR-10 & SVHN.ipynb** to run the Vision transformer models on Adversarial Attacks
* The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images
* SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with the minimal requirement on data formatting but comes from a significantly harder, unsolved, real-world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images

# Requirements =
Run pip install requirement.txt to install all requrements!


# Below code runs the Vision Transformer models on CIFAR-10 and SVHN datasets

cifar_trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)
cifar_trainer.train()

svhn_trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)
svhn_trainer.train(



Acknowlegements:
This repository is built upon the following four repositories:
https://github.com/yaodongyu/TRADES
https://github.com/YisenWang/MART
https://github.com/rwightman/pytorch-image-models
https://github.com/RulinShao/on-the-adversarial-robustness-of-visual-transformer
