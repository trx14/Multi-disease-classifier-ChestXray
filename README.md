# Multi-disease-classifier-ChestXray
## Abstract
Our goal is to investigate using only 'case-level' labels extracted automatically from radiology reports to construct a multi-disease classifier for CT scans with deep learning method. We chose three lung diseases as a start: atelectasis, pulmonary edema, and pneumonia. From a dataset of approximately 5,000 chest CT cases from our institution, we used a rule-based model to analyze those radiologist reports, labeling disease by text mining to identify cases with those diseases. From those results, we randomly selected the following mix of cases: 140 normal, 180 atelectasis, 195 pulmonary edema, and 190 pneumonia.  

As a key feature of this study, each chest CT scan was represented by only 10 axial slices (taken at regular intervals through the lungs), and furthermore all slices shared the same label based on the radiology report. So the label was weak, because often disease will not appear in all slices. We used ResNet-50[1] as our classification model, with 4-fold crossvalidation. Each slice was analyzed separately to yield a slice-level performance. For each case, we chose the 5 slices with highest probability and used their mean probability as the final patient-level probability. Performance was evaluated using the receiver operating characteristic (ROC) area under the curve (AUC). First for each of the three disease groups against the normal group, then for all the diseases combined as “abnormal” against the normal group. For the 3 diseases separately, the slice-based AUCs were 0.80 for atelectasis, 0.95 for edema, and 0.91 for pneumonia. The patient-based AUC were 0.85 for atelectasis, 0.97 for edema, and 0.95 for pneumonia. When the diseases were combined, the overall AUC was 0.90 for slice-based and 0.94 for patient-based classification.  

We backprojected the activations of last convolution layer and the weights from prediction layer to synthesize a heat map[2]. This heat map could be an approximate disease detector, also could tell us feature patterns which ResNet-50 focus on.  

### Reference
[1] He, Kaiming et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 770-778.  
[2] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva and A. Torralba, "Learning Deep Features for Discriminative Localization," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 2921-2929.  
