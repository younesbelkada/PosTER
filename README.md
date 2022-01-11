# PosTER
Implementation of PosTER - Pose Transformer Encoder Representation - for pedestrian attributes recognition 

## First steps

+ Decide on which pedestrian attribute to work on (chose 2-3) [JAAD Dataset](https://github.com/ykotseruba/JAAD) and [TITAN Dataset](https://usa.honda-ri.com/titan)
+ Reproduce some results with SOTA methods for these attributes
+ Train PosTER on unlabeled human poses (start with poses from PIE [here](https://drive.google.com/file/d/195g6eDeAaLRt7nEN5EweB7-eWwbktkQ_/view))
+ Fine-tune PosTER on these attributes


- Demander resultats sur merge_cls
- solution pour les autres classes?
- force complete pose?
- Is Focal Loss really useful?
- TCG dataset input format

## TO DO

- [ ] Relative coordinates and inflate => + data augmentation (flip and random translation / random scaling)
- [x] Import MonoLoco
- [x] Data augmentation
- [x] F1 score per class