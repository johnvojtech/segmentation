#!/bin/bash
ls data_onmt|grep cz_sk|grep -v small |xargs -I % python3 segment_onmt.py train --src data_onmt/%/data_%.train.src --tgt  data_onmt/%/data_%.train.tgt --src_vocab data_onmt/%/data_%.vocab.src --tgt_vocab data_onmt/%/data_%.vocab.tgt --model_dir onmt_models/% --src_val data_onmt/%/data_%.dev.src --tgt_val data_onmt/%/data_%.dev.tgt

#ls data_onmt|xargs -I % python3 segment_onmt.py translate --src data_onmt/%/data_%.test.src --src_vocab data_onmt/%/data_%.vocab.src --tgt_vocab data_onmt/%/data_%.vocab.tgt --model_dir onmt_models_cz_el/% > results_cz.tsv
