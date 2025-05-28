from add_path import add_project_path
add_project_path()

from context_loader_trt import load_tensorrt_classifier, load_tensorrt_classifier_engine, dynamic_context_trt, context_label_trt 

engine, context = load_tensorrt_classifier_engine()

engine_2, context_2, english_dictionary, english_list, word_alias_dictionary, train_categeory_data_json = load_tensorrt_classifier()

question = "Can I edit my profile identity ?"

possibility_predicted, label, inference_time = context_label_trt(question, english_dictionary, engine, context, train_categeory_data_json)

print(possibility_predicted, label, inference_time)