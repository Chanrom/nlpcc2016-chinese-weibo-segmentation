# nlpcc2016-chinese-weibo-segmentation

Long Short-Term Memory (LSTM) based model for the NLPCC 2016 shared task - [Chinese Weibo word segmentation](https://github.com/FudanNLP/NLPCC-WordSeg-Weibo). The model got 1st place in close and semi-open track. For more details on the model, refer to our paper [Recurrent Neural Word Segmentation with Tag Inference](https://link.springer.com/chapter/10.1007/978-3-319-50496-4_66)

## Requirement
theano
lasagne

## Notes
* The original dataset for this task should be requested by filling up a [Agreement Form](https://github.com/FudanNLP/NLPCC-WordSeg-Weibo/blob/master/FDU_agreement_form.pdf). So here we only provide a few examples.
* Once the original dataset is obtained, one should change the space-splited format to BMES tagging format.
* To get the unsupervised features, use scripts by Wu et al., 2014 [CistSegment](https://github.com/wugh/CistSegment)

## Citation
If you use this software, please cite our paper.
```
@InProceedings{zhou2016lstmtaginference,
  Title                    = {ORecurrent neural word segmentation with tag inference},
  Author                   = {Qianrong Zhou, Long Ma, Zhenyu Zheng, Yue Wang, and Xiaojie Wang},
  Booktitle                = {Proceedings of The Fifth Conference on Natural Language Processing and Chinese Computing \& The Twenty Fourth
International Conference on Computer Processing of Oriental Languages},
  Year                     = {2016}
}
```
