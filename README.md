# kg-goal-oriented-dialog
This code is written in Python and implements a goal-oriented dialog system which takes as input a conversation history as well as the underlying database, and predicts the best next utterance. 

# Get started
## Prepare the data

### Build data
`python build_all_data.py -train_path $path_to_train_data -dev_path $path_to_dev_data -db_path $path_to_database -out_dir $path_to_out_dir [-test_path $path_to_test_data]`

### Dump pretrained word embeddings
`python build_pretrained_w2v.py -emb $path_to_pretrained_word_embeddings -vocab_path $path_to_vocabulary -emb_size $embedding_size -out_path $path_to_output_file`

## Train the model
`python train.py -vocab_size $vocab_size -mf $path_to_saved_model -data_dir $path_to_data_dir -pre_w2v  $pretrained_word_embeddings -hidden_size $hidden_size -word_emb_size $word_emb_size -model_name $model_name -eval_every $evaluate_model_every_steps -exp $experiment_index`

## Evaluate the model
`python test.py -data_dir $path_to_data_dir -mf $path_to_saved_model -act $action_type`
To compute evaluation metrics, set $action_type as `evaluate`; To interpret the model (e.g., sample test dialogs with intermediate model attentions), set $action_type$ as `interpret`.
