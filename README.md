# exphypo


This is the code base for the paper "CNNs Reveal the Computational Implausibility of the Expertise Hypothesis" published in iScience (https://www.cell.com/iscience/fulltext/S2589-0042(23)00053-6).

The folder 'transfer' contains the code to fine-tune the face-trained, the object-trained and the dual-task trained model on the car task.

The folder 'eval_no_lesioning' contains the code to evaluate the performance fine-tuned models, and the 'eval_lesioning' folder contains the code to lesion the filters in the last conv layer of the dual-task CNN and to evaluate its performance.
