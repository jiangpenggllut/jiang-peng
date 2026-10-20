# JiangPeng



## Environments

* Python 3.9.7
 
* pytorch 1.10.2

* transformers 4.17.0

* datasets 2.0.0

* RTX Titan RTX

* CUDA 11.4


## Data

The URL for the RTE dataset is https://huggingface.co/datasets/glue (accessed on 25 December 2024). The URL for the MRPC dataset is http://research.microsoft.com/research/nlp/msr_paraphrase.htm (accessed on 21 August 2024). The URL for the SciTail dataset is https://opendatalab.org.cn/OpenDataLab/SciTail (accessed on 21 August 2024). The URL for the Medical-SM dataset is https://huggingface.co/datasets/jiangpeng852456/Medical-sm/tree/main  (accessed on 22 May 2026). The URL for the KUAKE-QTR dataset is https://tianchi.aliyun.com/dataset/95414 (accessed on 27 March 2025).

## Usage

* Download raw datasets from the above data links and put them into the directory **raw_data** like this:

	```
	--- raw_data
	  |
	  |--- medical
	  |--- KUAKE-QTR

          |--- mrpc
          |--- rte
          |--- scitail
	```

* We have tried various pre-trained models. The following models work fine with our code:

    * model names for RTE:
        - ALBERT
        - EntityCS
        - KBioXLM
        - XLM-R-base
        - Twitter-twhin-bert-base
        - BERT-base
        - DeBERTa-base

    * model names for MRPC :
        - DisRoBERTa
        - ALBERT
        - EntityCS
        - KBioXLM
        - XLM-R-base
        - funnel-transformer-medium
        - Twitter-twhin-bert-base
        - BERT-base
        - DeBERTa-base

     * model names for Scitail :
        - ALBERT
        - EntityCS
        - KBioXLM
        - XLM-R-base
        - funnel-transformer-medium
        - Twitter-twhin-bert-base
        - BERT-base
        - DeBERTa-base
        

    * model names for Medical:
        - hfl/chinese-macbert-base
        - hfl/chinese-roberta-wwm-ext
        - Twitter-twhin-bert-base
        - XLM-R-base
        - hongyin-bert-chinese-homie-base
        - KBioXLM
        - huawei-noah-EntityCS-39-PEP_MS_MLM-xlmr-base
        
     * model names for KUAKE-QTR :
        - hongyin-bert-chinese-homie-base
        - Twitter-twhin-bert-base
        - KBioXLM
        - huawei-noah-EntityCS-39-PEP_MS_MLM-xlmr-base
        - XLM-R-base
        - hfl/chinese-macbert-base
        - hfl/chinese-roberta-wwm-ext
        

* Pre-process datasets.

    ```
    PYTHONPATH=. python ./src/preprocess.py -raw_path raw_data/mrpc
    ```
    ```
    PYTHONPATH=. python ./src/preprocess.py -raw_path raw_data/rte
    ```
    ```
    PYTHONPATH=. python ./src/preprocess.py -raw_path raw_data/scitail
    ```
    ```
    PYTHONPATH=. python ./src/preprocess.py -raw_path raw_data/medical
    ```
    ```
    PYTHONPATH=. python ./src/preprocess.py -raw_path raw_data/KUAKE-QTR
    ```

* Training and Evaluation (Baseline)

    * MRPC
    ```
    PYTHONPATH=. python -u src/main.py \
        -baseline \
        -task mrpc \
        -model bert-base-uncased \
        -num_labels 2 \
        -batch_size 16 \
        -accum_count 1 \
        -test_batch_size 128 \
        >> logs/mrpc.bert-base-uncased.baseline.log

    ```
 
    * RTE
    ```
    PYTHONPATH=. python -u src/main.py \
        -baseline \
        -task rte \
        -model bert-base-uncased \
        -num_labels 2 \
        -batch_size 16 \
        -accum_count 1 \
        -test_batch_size 128 \
        >> logs/mrpc.bert-base-uncased.baseline.log

    ```

   * Scitail
    ```
    PYTHONPATH=. python -u src/main.py \
        -baseline \
        -task scitail \
        -model bert-base-uncased \
        -num_labels 2 \
        -batch_size 16 \
        -accum_count 1 \
        -test_batch_size 128 \
        >> logs/scitail.bert-base-uncased.baseline.log

    ```

    * Medical

    ```
  PYTHONPATH=. python -u src/main.py \
        -baseline \
        -task medical \
        -model hfl/chinese-roberta-wwm-ext \
        -num_labels 2 \
        -batch_size 16 \
        -accum_count 4 \
        -test_batch_size 128 \
        >> logs/medical.chinese-roberta-wwm-ext.baseline.log
    ```

* Training and Evaluation (JiangPeng)

    * RTE
    ```

  PYTHONPATH=. python -u src/main.py \
        -task mrpc \
        -model ./bert-base-uncased \
        -num_labels 2 \
        -batch_size 16 \
        -accum_count 1 \
        -test_batch_size 128 \
        >> logs/RTE_cls_loss_weight_rev_loss_weight_dbma_attention_keyword_chapter4_revloss_keyword_loss_kl_6_bert-base-uncased.log
    ```

    * MRPC
    ```

  PYTHONPATH=. python -u src/main.py \
        -task mrpc \
        -model ./bert-base-uncased \
        -num_labels 2 \
        -batch_size 16 \
        -accum_count 1 \
        -test_batch_size 128 \
        >> logs/MRPC_cls_loss_weight_rev_loss_weight_dbma_attention_keyword_chapter4_revloss_keyword_loss_kl_6_bert-base-uncased.log
    ```

    * Scitail
   ```

  PYTHONPATH=. python -u src/main.py \
        -task mrpc \
        -model ./bert-base-uncased \
        -num_labels 2 \
        -batch_size 16 \
        -accum_count 1 \
        -test_batch_size 128 \
        >> logs/scitail_cls_loss_weight_rev_loss_weight_dbma_attention_keyword_chapter4_revloss_keyword_loss_kl_6_bert-base-uncased.log
    ```

   * Medical
    ```
  PYTHONPATH=. python -u src/main.py \
        -task medical \
        -model ./bert-base-uncased \
        -num_labels 3 \
        -batch_size 16 \
        -accum_count 4 \
        -test_batch_size 128 \
        >> logs/Medical_cls_loss_weight_rev_loss_weight_dbma_attention_keyword_chapter4_revloss_keyword_loss_kl_6_bert-base-uncased.log
   ```

   * KUAKE-QTR
    ```
  PYTHONPATH=. python -u src/main.py \
        -task KUAKE-QTR \
        -model ./bert-base-uncased \
        -num_labels 4 \
        -batch_size 16 \
        -accum_count 4 \
        -test_batch_size 128 \
        >> logs/KUAKE-QTR_cls_loss_weight_rev_loss_weight_dbma_attention_keyword_chapter4_revloss_keyword_loss_kl_6_bert-base-uncased.log
    ```                                                                                                                                                                                                                                   
