# exphypo


This is the code base for the paper "CNNs Reveal the Computational Implausibility of the Expertise Hypothesis" published in iScience (https://www.cell.com/iscience/fulltext/S2589-0042(23)00053-6).

The folder 'transfer' contains the code to fine-tuned the face-trained, the object-trained and the dual-task trained model to discriminate cars.

The folder 'eval_no_lesioning' contains the code to evaluate the performance fine-tuned models, and the 'eval_lesioning' folder contains the code to lesion the filters in the last conv layer and to evaluate the performance.
