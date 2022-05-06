For saving predictions from Trankit:
 - modify  `~/miniconda3/envs/ner1/lib/python3.8/site-packages/trankit/tpipeline.py`
   copy this at the end of `_eval_ner` function:
   ```        
   import pickle
   with open("/home/s/suba13/Work/NER/my_dataset/trankit/predictions", "wb") as fp:
   pickle.dump(predictions, fp)
   ```