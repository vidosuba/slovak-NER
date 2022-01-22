# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run.sh

#--dataset_name conll2003 \

python3 run_ner.py \
  --model_name_or_path gerulata/slovakbert \
  --train_file ../data/wikiann_cleaned_data/train_cleaned_hg.json \
  --validation_file ../data/wikiann_cleaned_data/dev_cleaned_hg.json \
  --test_file ../data/wikiann_cleaned_data/test_cleaned_hg.json \
  --text_column_name sentence \
  --label_column_name word_labels \
  --output_dir ./output/test-ner \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --num_train_epochs 13 \
  --evaluation_strategy epoch \
  --save_strategy no \
  --do_predict


#python3 run_ner.py \
#  --model_name_or_path gerulata/slovakbert \
#  --train_file ../data/wikiann_cleaned_data/train_cleaned_hg.json \
#  --validation_file ../data/wikiann_cleaned_data/dev_cleaned_hg.json \
#  --test_file ../data/wikiann_cleaned_data/test_cleaned_hg.json \
#  --text_column_name sentence \
#  --label_column_name word_labels \
#  --output_dir ./output/test-ner2 \
#  --overwrite_output_dir \
#  --do_train \
#  --do_eval \
#  --num_train_epochs 10 \
#  --max_seq_length 512 \
#  --per_gpu_train_batch_size 32 \
#  --evaluation_strategy epoch \
#  --save_strategy no \


