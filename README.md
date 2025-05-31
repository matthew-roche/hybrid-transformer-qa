# Hybrid Question Answering with Transformers  
A dual model system for answering questions based on business context. Scalable across lengthier contexts.

The way this system functions is, 
1. Business context is first categorized into features like "feature X can do", "feature X can't do" and capabilities as sentences.    
2. Then the classifier model is trained on this dataset to learns to pay attention in categorizing "feature X" and it's possiblity "can do"/"can't do".       
3. Then the trained category predicted is selected to load the feature capability sentences.   
4. The capability sentences and the question is passed to DeBERTa v3 to extract the relevant answer.   
5. Possibility (Yes/No) + answer is finally combined together.   

Benefits of this approach:   
1. Large contexts are broken to smaller categorized contexts = scalability, better understanding of data.   
2. Classifier predicting context category supports DeBERTa v3 to quickly load the context data.   
3. Trained categorized context reused for question answering.   
4. Classifier is trained on context only, therefore questions asked does not need to be trained separately.   
5. Not fully dependent on DeBERTa v3, and another answer extraction model like RoBERTa can be used instead.   
6. Sanitizing question to work on trained vocabulary only.    

Refer [Model Simplified Flowchart](https://) for more details on how the 2 models work together.

This project contains the dataset used to train, in directory ```/data/cat2context_train.json```, this dataset can be modified to fit the business context.   
And the test dataset is in ```/data/cat2context_test.json```, this contains questions to see how well the model has learned the category and possiblity.   

This is the project structure:   
```
├───api
│       model_blueprint.py
│       model_init.py
│       model_schema.py
│
├───data
│       cat2context_test.json
│       cat2context_train.json
│       context_mapper.json
│       dataset_loader.py
│       deberta_train.json
|       inference_test.json
│       word_alias_mapper.json
│
├───docs
├───fine_tuned_models
├───huggingface_model
│
├───model
│       classifier_model.py
│
├───output
│   └───models
│       │   classifier_591_68_75.pt
│       │   classifier_597_63_63.pt
│       │   english_dictionary_591_68_75.json
│       │   english_dictionary_597_63_63.json
│       │   english_list_591_68_75.json
│       │   english_list_597_63_63.json
│       │   training_curves_591_68_75.json
│       │   training_curves_597_63_63.json
│       │
│       ├───onnx
│       │       .gitignore
│       │       classifier_597_63_63.onnx
│       │
│       └───tensorrt
│               classifier_597_63_63-4.trt
│               classifier_597_63_63-5.trt
│
scripts
│       add_path.py
│       analyze_classifier.py
│       classifier_to_onnx.py
│       classifier_to_trt.py
│       context_loader.py
│       context_loader_trt.py
│       deberta_finetune.py
│       dev_api.py
│       inference.py
│       inference_onnx.py
│       inference_trt.py
│       qa_to_onnx.py
│       qa_to_trt.py
│       train_classifier.py
│    
└───util
        helper.py
```

## Getting started guide   
Based on Python version [3.13.2](https://www.python.org/downloads/release/python-3132/)   

This repository includes a backend API for continous inference and testing purposes, after the steps from 1 to 4 below are completed, skip to "Dev API Inference" section.    

### Installation, Analysis & Inference

Packages used: [PyTorch v2.7 + cuda 12.8](https://github.com/pytorch/pytorch/releases/tag/v2.7.0), [matplotlib](https://pypi.org/project/matplotlib/), [Huggingface transformers](https://pypi.org/project/transformers/), [Huggingface datasets](https://pypi.org/project/datasets/), [Flask](https://pypi.org/project/Flask/), [Flask-smorest for swagger-ui](https://flask-smorest.readthedocs.io/en/latest/)   

#### STEP 1
It is recommended to use a virtual python environment, this can be done with;
```
python -m venv dl
```

#### STEP 2
Then on windows, the virtual env can be activated by 
```
.\dl\Scripts\activate
```   

#### STEP 3
Afterwards, Install the packages from ```requirements.txt```, can be done using;
```
pip install -r <project-dir>/requirements.txt
```

##### TensorRT SDK for inference, and optimization
After installing the packages, download the appropiate tensorRT SDK from https://developer.nvidia.com/tensorrt. This project was built using TensorRT 10.11.0.33 version.    

After downloaded, copy the contents to a folder accessible to the python enviornment, then install the tensorrt-<version>-cp<python>-none-<os>-<arch>.whl from the python folder inside the contents.
For instance,
```
pip install TensorRT-10.11.0.33/python/tensorrt-10.11.0.33-cp313-none-win_amd64
```
Other types such as dispatch can also be used if the motive is to inference and not to optimize the engine.   

#### STEP 4
```
python <project-dir>/scripts/deberta_finetune.py
``` 
This will download DeBERTa v3 base squad2 trained by deepset from hugging face and save locally, and then it will fine tune. The fine tuning is done using the dataset in ```/data/deberta_train.json```   

- Note: Any pretrained answer extraction model from hugging face can be used (such as RoBERTa) by changing the ```model_name``` variable in ```/scripts/deberta_finetune.py```  

This execution will save the hugging face model under ```/huggingface_model``` directory and the fine tuned model under ```/fine_tuned_models``` directory, these are needed for later steps.   

#### STEP 5
Now that we have the hugging face model, and a trained classifier is provided in this repository under ```/output/models/```. So we will see how the classifier and the hugging face model works together.

Run 
```
python <project-dir>/scripts/analyze_classifier.py
```
This will load both DeBERTa v3 and classifier models to run inference against the test dataset, and at the end it will show a series of graphs to indicate how the trained classifier model is performing on unseen test data. These graphs are included under ```docs``` in this repository as well.   

[Training Loss curve, across epochs](https://)   
[Test dataset accuracy, across epochs](https://)   
[Category Softmax curves](https://)   
[Possibility Softmax curves](https://)   

#### STEP 6
After the classifier and DeBERTa functionality is analyzed, now run the command below to test how the fine tuned model performs with the classifier.
```
python <project-dir>/scripts/inference.py
``` 
The questions included in this file can be modified as needed, the more failure conditions occur the better understanding can be gained.   

### Dev API Inference (requires steps 1 to 4 from above)

The API can be hosted by the command;
```
python <project-dir>/scripts/dev_api.py
```
By default port 5000 will be used and this server can be accessed via ```http://127.0.0.1:5000/api/docs```

The ```api``` folder in the repository holds the blueprint which interacts with the models, and the marhmallow schema for input validation and output structuring.

### Training the classifier

The code for this is in ```scripts/train_classifier.py``` here the train and test datasets are loaded, train datset is split to 80 batches, and iteratively (epochs) model weights are updated via adamW optimizer.
This process is done multiple times to pick the best model, this can be adjusted by modifying the ```num_of_runs``` variable in line 148. Similarly learn rate, epochs, and other params can be modified too, the best values are the default.   

The Classifier model is instantiated in line 163, context label mapper can be created by uncommenting lines 44 to 47, in this case category_count is 6   
```python
model = Classifier(max_vocab_size=len(word_list), embed_dim=512, category_count=len(context_label_mapper), possibility_count=2).to(device) # copy model to torch device
```

Training can be done by executing,   
```
python <project-dir>/scripts/train_classifier.py
```

### Optimizing multimodel for inference (Onnx, TensorRT)
The process here is torch model to onnx model to tensorrt engine.   

| Files in /Scripts | Purpose |
| ------------- | ------------- |
| classifier_to_onnx.py | Loads the torch classifier model, converts to onnx, then asserts the onnx model inference against torch model. |
| classifier_to_trt.py  | Loads the onnx classifier model, converts to tensorrt engine with optimization profile 5 and max shape (8, 64). Means 8 batches with 64 max size. |
| qa_to_onnx.py | Loads the fine tuned DeBeRtav3 model, converts to onnx, then inferences on onnx runtime to ensure it functions.  |
| qa_to_trt.py | This will be released after [TensorRT Issue#4288](https://github.com/NVIDIA/TensorRT/issues/4288#issue) is resolved. |
| inference_onnx.py | Test onnx performance. Combines onnx classifier model and onnx finetuned DeBeRtaV3 for hybrid inference. |
| inference_trt.py | Test trt performance. Combines tensorrt classifier engine and torch(cuda) finetuned DeBeRtaV3 for hybrid inference |
| inference.py | Test raw performance. Combines torch(cuda) classifier engine and torch(cuda) finetuned DeBeRtaV3 for hybrid inference |

### Benchmark   

Execution Computer specifications:   
| Component     | Specification |
| ------------- | ------------- |
| CPU | i5 12600K |
| RAM | 32GB DDR4 |
| Storage | nvme ssd |
| GPU | RTX 4080 super |

#### Results, tested against the questions in /data/inference_test.json   

| File | Run 1 | Run 2 | Run 3 |
| -----| ----- | ----- | ----- |
| inference.py | 14.425 | 14.527 | 14.408 |
| inference_onnx.py | 28.767 | 28.518 | 28.990 |
| inference_trt.py | 14.252 | 14.363 | 14.331 |

Concludes that the classifier tensorRT engine performs 70 to 170ms faster than with torch(cuda), this is important when combined with the Flask API, especially for batch inference.   

More Updates and Benchnmarks coming soon...   

## Acknowledgments

- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)   
- [Huggingface DeBERTa v3 base squad2 from deepset](https://huggingface.co/deepset/deberta-v3-base-squad2)   
- [Huggingface Trainer Documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer)
- [matplotlib Documentation](https://matplotlib.org/stable/tutorials/pyplot.html)   
- Developed as part of the MIT xPRO course: *Deep Learning – Mastering Neural Networks (2025 Feb Cohort)*
