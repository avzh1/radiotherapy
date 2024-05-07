Idea as inspired by In [23] of the following jupyter notebook from SAM:
https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb

In SAM there is an ability to provide a selection of points to exclude from the search
area at the same time, we can provide a bounding box for the containing structure.

Therefore, we can run a model that will learn the bounding box coodrinates for the
segentation organ, and also, it will predict some exclusion points. This may help for
making sure that the organs stay separate to eachother in the organ logic.

Pipeline would look something like:

- Run a model to predict the bounding box. We know for sure that the organ is somewhere
  here. This can be extracted possibly from the nnUNet-based architecture.
- Simultaneously, we write a model that will predict a set of points within the bounding
  box, where we know 100% that the organ doesn't segment there. We need to make sure that
  there are no False Negatives there becuase the SAM model will attempt to exclude this
  area. 
- Run a finetuned version of SAM inference which will draw a square around the organ, and
  provide a set of exclusion points. This may be a hyperparameter we need to fix and
  see...
- The final result should account for the area which is for sure the organ, and also learn
  to exclude some areas because of overlap with other segmentations as specified by the
  clinical team.