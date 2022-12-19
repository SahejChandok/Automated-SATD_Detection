# CSI5137-Automated_SATD_Detection
Final project for CSI 5137 at uOttawa. Comparative study on the use of ML to detect SATD in source code comments, with a focus on PTMs.

Uses the dataset from https://github.com/maldonado/tse.satd.data. 

Tested on SVM, Logistic Regression, CNN, LSTM, Roberta, Deberta V1 and V3, and Sbert. Base models were used for all transformers. Transformers were loaded from HuggingFace and submitted as jobs to a remote cluster, with the output and results stored in the "transformer results" folder. Note that the models were loaded locally on the remote server due to the cluster's inability to download them, so the from_pretrained calls will have to be changed to load them from HuggingFace.

Results for the traditional classifiers and NNs are in their respective scripts in the "training scripts" folder.

Saved CNN and LSTM models are in the "saved models" directory as they were created in Tensorflow which doesn't allow proper seeding for reproducability. The saved transformer models were too big to be placed on GitHub, and the traditional models were not saved as they are easy to rerun.

We also ran a quick test using our best transformer model (Deberta V3 with weighted CE loss) on the SATD portion of the R dataset from Sharma et al. (https://link.springer.com/article/10.1007/s10515-022-00358-6), available at: https://zenodo.org/record/4558220. Its results are in the file R_test.ipynb in the "training scripts" folder.
