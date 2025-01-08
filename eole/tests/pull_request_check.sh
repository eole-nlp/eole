#!/bin/bash
# Run this script and fix *any* error before sending PR.
# For repeated runs, set the environment variables
# SKIP_DOWNLOADS=1  If files/uncompressed dirs exist don't download (if compressed files exist, just untar).
# SKIP_FULL_CLEAN=1  Don't remove anything downloaded/uncompressed.

SKIP_FULL_CLEAN=0

PROJECT_ROOT=`dirname "$0"`"/../.."
DATA_DIR="$PROJECT_ROOT/eole/tests/data"
TEST_DIR="$PROJECT_ROOT/eole/tests"
PYTHON="python3"
# TMP_OUT_DIR="/tmp/eole_prchk"
TMP_OUT_DIR=$PROJECT_ROOT/tests_outputs
mkdir -p $TMP_OUT_DIR
# LOG_FILE=/tmp/$$_pull_request_chk.log
LOG_FILE=$TMP_OUT_DIR/$$_pull_request_chk.log
echo > ${LOG_FILE} # Empty the log file.

PYTHON=python3


clean_up()
{
    # if [[ "$1" != "error" ]]; then
    #     rm ${LOG_FILE}
    # fi
    if [[ "${SKIP_FULL_CLEAN}" == "1" ]]; then
        # delete any .model's that weren't downloaded
        ls $TMP_OUT_DIR/*.model | xargs -I {} rm -rf $TMP_OUT_DIR/{}
    else
        rm -rf $TMP_OUT_DIR/dump_pred
        rm -rf $TMP_OUT_DIR/*.model
        rm -rf $TMP_OUT_DIR/eole.train.check
        rm -f $TMP_OUT_DIR/eole.vocab.*
    fi
}
trap clean_up SIGINT SIGQUIT SIGKILL

error_exit()
{
    echo "Failed !" | tee -a ${LOG_FILE}
    echo "[!] Check ${LOG_FILE} for detail."
    clean_up error
    exit 1
}

# environment_prepare()
# {

# }

# black check
echo -n "[+] Doing Black check..."
${PYTHON} -m black --check . >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

# flake8 check
echo -n "[+] Doing flake8 check..."
#${PYTHON} -m flake8 --ignore *venv* . >> ${LOG_FILE} 2>&1
${PYTHON} -m pflake8 . >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

# unittest
echo -n "[+] Doing unittest test..."
${PYTHON} -m unittest discover >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

# Make sure recipes configs are valid
echo -n "[+] Checking recipes config..."
${PYTHON} eole/tests/test_recipes.py $PROJECT_ROOT/recipes

# Get Vocabulary test
echo -n "[+] Testing vocabulary building..."
PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH} ${PYTHON} eole/bin/main.py build_vocab \
            -config ${DATA_DIR}/data.yaml \
            -save_data $TMP_OUT_DIR/eole \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.tgt \
            -n_sample 5000 -overwrite >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -f -r $TMP_OUT_DIR/sample

#
# Training test
#
echo -n "[+] Testing architecture rnn sample dump..."
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/data.yaml \
            -model '{"architecture": "rnn"}' \
            -save_data $TMP_OUT_DIR/eole.train.check \
            -n_sample 30 \
            -overwrite \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
# rm $TMP_OUT_DIR/eole.train.check*  # used in tool testing

echo "[+] Doing Training test..."

echo -n "  [+] Testing architecture rnn training..."
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -model '{"architecture": "rnn", "hidden_size": 10, "embeddings": {"word_vec_size": 5, "position_encoding_type": None}}' \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 10}' \
            -report_every 5 \
            -tensorboard \
            -tensorboard_log_dir $TMP_OUT_DIR/logs_train >> ${LOG_FILE} 2>&1
${PYTHON} eole/tests/test_events.py --logdir $TMP_OUT_DIR/logs_train -tensorboard_checks train
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/logs_train

echo -n "  [+] Testing architecture rnn training and validation..."
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -model '{"architecture": "rnn", "hidden_size": 10, "embeddings": {"word_vec_size": 5, "position_encoding_type": None}}' \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 10, "valid_steps": 5}' \
            -report_every 2 \
            -tensorboard \
            -tensorboard_log_dir $TMP_OUT_DIR/logs_train_and_valid \
            >> ${LOG_FILE} 2>&1
${PYTHON} eole/tests/test_events.py --logdir $TMP_OUT_DIR/logs_train_and_valid -tensorboard_checks train
${PYTHON} eole/tests/test_events.py --logdir $TMP_OUT_DIR/logs_train_and_valid -tensorboard_checks valid
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/logs_train_and_valid

echo -n "  [+] Testing architecture rnn training w/ coverage..."
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -model '{"architecture": "rnn", "hidden_size": 10, "embeddings": {"word_vec_size": 5, "position_encoding_type": None}, "decoder": {"coverage_attn": True, "lambda_coverage": 0.1}}' \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 10}' \
            -report_every 5 \
            >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing architecture custom transformer training w/ align..."
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/align_data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -model '{"layers": 4, "hidden_size": 16, "transformer_ff": 64, "embeddings": {"word_vec_size": 16, "position_encoding_type": None}, "encoder": {"encoder_type": "transformer", "heads": 2}, "decoder": {"decoder_type": "transformer", "lambda_align": 0.05, "alignment_layer": 2, "alignment_heads": 0, "heads": 2}}' \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 10}' \
            -report_every 5 \
            >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing architecture custom transformer training w/ validation with dynamic scoring..."
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -model '{"layers": 4, "hidden_size": 16, "transformer_ff": 16, "embeddings": {"word_vec_size": 16, "position_encoding_type": "SinusoidalInterleaved"}, "encoder": {"encoder_type": "transformer", "heads": 2}, "decoder": {"decoder_type": "transformer", "heads": 2,}}' \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 10, "valid_steps": 5}' \
            -report_every 2 \
            -valid_metrics "BLEU" "TER" \
            -tensorboard \
            -scoring_debug \
            -dump_preds $TMP_OUT_DIR/dump_pred \
            -tensorboard_log_dir $TMP_OUT_DIR/logs_dynamic-scoring >> ${LOG_FILE} 2>&1
      
${PYTHON} eole/tests/test_events.py --logdir $TMP_OUT_DIR/logs_dynamic-scoring -tensorboard_checks valid_metrics
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/logs_dynamic-scoring

echo -n "  [+] Testing architecture transformer training w/ validation with dynamic scoring and maxrelative ..."
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -model '{"architecture": "transformer", "layers": 4, "hidden_size": 16, "transformer_ff": 64, "heads": 2, "share_decoder_embeddings": True, "share_embeddings": True, "embeddings": {"word_vec_size": 16, "position_encoding_type": "Relative", "n_positions": 8}}' \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 10, "valid_steps": 5}' \
            -valid_metrics "BLEU" "TER" \
            -report_every 2 \
            -tensorboard \
            -scoring_debug \
            -dump_preds $TMP_OUT_DIR/dump_pred \
            -tensorboard_log_dir $TMP_OUT_DIR/logs_dynamic-scoring_and_relative >> ${LOG_FILE} 2>&1
      
${PYTHON} eole/tests/test_events.py --logdir $TMP_OUT_DIR/logs_dynamic-scoring_and_relative -tensorboard_checks valid_metrics
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/logs_dynamic-scoring_and_relative

echo -n "  [+] Testing architecture transformer training w/ validation with dynamic scoring and rotary ..."
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -model '{"architecture": "transformer", "layers": 4, "hidden_size": 16, "transformer_ff": 64, "heads": 2, "encoder": {"encoder_type": "transformer"}, "embeddings": {"word_vec_size": 16, "position_encoding_type": "Rotary"}}' \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 10, "valid_steps": 5}' \
            -valid_metrics "BLEU" "TER" \
            -report_every 2 \
            -tensorboard \
            -scoring_debug \
            -dump_preds $TMP_OUT_DIR/dump_pred \
            -tensorboard_log_dir $TMP_OUT_DIR/logs_dynamic-scoring_and_rotary >> ${LOG_FILE} 2>&1

${PYTHON} eole/tests/test_events.py --logdir $TMP_OUT_DIR/logs_dynamic-scoring_and_rotary -tensorboard_checks valid_metrics
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/logs_dynamic-scoring_and_rotary

echo -n "  [+] Testing architecture transformer training w/ validation with dynamic scoring and alibi ..."
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.tgt \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            -model '{"architecture": "transformer", "layers": 4, "hidden_size": 16, "transformer_ff": 64, "heads": 2, "embeddings": {"word_vec_size": 16, "position_encoding_type": "Alibi"}}' \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 10, "valid_steps": 5}' \
            -valid_metrics "BLEU" "TER" \
            -report_every 2 \
            -tensorboard \
            -scoring_debug \
            -dump_preds $TMP_OUT_DIR/dump_pred \
            -tensorboard_log_dir $TMP_OUT_DIR/logs_dynamic-scoring_and_alibi >> ${LOG_FILE} 2>&1
      
${PYTHON} eole/tests/test_events.py --logdir $TMP_OUT_DIR/logs_dynamic-scoring_and_alibi -tensorboard_checks valid_metrics
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm -r $TMP_OUT_DIR/logs_dynamic-scoring_and_alibi

echo -n "  [+] Testing architecture custom decoder only training..."
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/lm_data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -share_vocab \
            -model '{"hidden_size": 16, "transformer_ff": 64, "embeddings": {"word_vec_size": 16}, "encoder": None, "decoder": {"decoder_type": "transformer", "layers": 2, "heads": 4}}' \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 10}' \
            -report_every 5 \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            >> ${LOG_FILE} 2>&1
            # -tgt_vocab $TMP_OUT_DIR/eole.vocab.src \
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

#
# Translation test
#
echo "[+] Doing translation test..."

echo -n "  [+] Testing RNN translation..."
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} eole/bin/main.py predict -model_path ${TEST_DIR}/test_model -src $TMP_OUT_DIR/src-test.txt -verbose >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt

echo -n "  [+] Testing RNN ensemble translation..."
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} eole/bin/main.py predict -model_path ${TEST_DIR}/test_model ${TEST_DIR}/test_model \
            -src $TMP_OUT_DIR/src-test.txt -verbose >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt

echo -n "  [+] Testing RNN translation w/ Beam search..."
${PYTHON} eole/bin/main.py predict -model_path ${TEST_DIR}/test_model2  \
            -src ${DATA_DIR}/morph/src.valid   \
            -verbose \
            -batch_size 10 \
            -beam_size 10 \
            -tgt ${DATA_DIR}/morph/tgt.valid   \
            -out $TMP_OUT_DIR/trans_beam  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/morph/tgt.valid $TMP_OUT_DIR/trans_beam
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/trans_beam

echo -n "  [+] Testing RNN translation w/ Random Sampling..."
${PYTHON} eole/bin/main.py predict -model_path ${TEST_DIR}/test_model2  \
            -src ${DATA_DIR}/morph/src.valid   \
            -verbose -batch_size 10     \
            -beam_size 1                \
            -seed 1                     \
            -top_k -1    \
            -temperature 0.0001    \
            -tgt ${DATA_DIR}/morph/tgt.valid   \
            -out $TMP_OUT_DIR/trans_sampling  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/morph/tgt.valid $TMP_OUT_DIR/trans_sampling
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/trans_sampling

echo -n "  [+] Testing LM generation..."
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} eole/bin/main.py predict -model_path ${TEST_DIR}/test_model_lm -src $TMP_OUT_DIR/src-test.txt -verbose >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt

echo -n "  [+] Testing LM generation w/ Beam search..."
${PYTHON} eole/bin/main.py predict -model_path ${TEST_DIR}/test_model_lm  \
            -src ${DATA_DIR}/data_lm/src-gen.txt   \
            -verbose -batch_size 1     \
            -beam_size 10 \
            -ban_unk_token \
            -length_penalty none \
            -out $TMP_OUT_DIR/gen_beam  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/data_lm/gen-beam-sol.txt $TMP_OUT_DIR/gen_beam
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/gen_beam

echo -n "  [+] Testing LM generation w/ Random Sampling..."
${PYTHON} eole/bin/main.py predict -model_path ${TEST_DIR}/test_model_lm  \
            -src ${DATA_DIR}/data_lm/src-gen.txt   \
            -verbose -batch_size 1     \
            -beam_size 1                \
            -seed 1                     \
            -top_k -1    \
            -temperature 0.0001    \
            -ban_unk_token \
            -length_penalty none \
            -out $TMP_OUT_DIR/gen_sampling  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/data_lm/gen-sampling-sol.txt $TMP_OUT_DIR/gen_sampling
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/gen_sampling

echo -n "  [+] Testing LM generation w/ Random Top-k/Nucleus Sampling..."
${PYTHON} eole/bin/main.py predict -model_path ${TEST_DIR}/test_model_lm  \
            -src ${DATA_DIR}/data_lm/src-gen.txt   \
            -verbose -batch_size 1     \
            -beam_size 1                \
            -seed 3                     \
            -top_k -1    \
            -top_p 0.95    \
            -temperature 1    \
            -ban_unk_token \
            -length_penalty none \
            -out $TMP_OUT_DIR/gen_sampling  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/data_lm/gen-nucleus-sampling-sol$($PYTHON -c "import torch; print(torch.__version__[0])").txt $TMP_OUT_DIR/gen_sampling
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/gen_sampling

echo -n "  [+] Testing LM generation w/ Random Top-k/Nucleus Sampling and multi beams..."
${PYTHON} eole/bin/main.py predict -model_path ${TEST_DIR}/test_model_lm  \
            -src ${DATA_DIR}/data_lm/src-gen.txt   \
            -verbose -batch_size 1     \
            -beam_size 10                \
            -seed 2                     \
            -top_k 50    \
            -top_p 0.95    \
            -temperature 1    \
            -length_penalty avg \
            -ban_unk_token \
            -min_length 5 \
            -out $TMP_OUT_DIR/gen_sampling  >> ${LOG_FILE} 2>&1
diff ${DATA_DIR}/data_lm/gen-sampling-beams-sol$($PYTHON -c "import torch; print(torch.__version__[0])").txt $TMP_OUT_DIR/gen_sampling
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/gen_sampling

#
# Inference engines test
#
echo -n "  [+] Testing PY LM inference engine .."
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} eole/tests/test_inference_engines.py -model ${TEST_DIR}/test_model_lm \
            -model_type decoder \
            -input_file $TMP_OUT_DIR/src-test.txt \
            -inference_config_file ${DATA_DIR}/inference-engine_py.yaml \
            -out $TMP_OUT_DIR/inference_engine_lm_py_outputs  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt
rm $TMP_OUT_DIR/inference_engine_lm_py_outputs_file.json
rm $TMP_OUT_DIR/inference_engine_lm_py_outputs_list.json

echo -n "  [+] Testing CT2 LM inference engine .."
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} eole/tests/test_inference_engines.py -model ${TEST_DIR} \
            -model_type decoder \
            -input_file $TMP_OUT_DIR/src-test.txt \
            -inference_config_file ${DATA_DIR}/inference-engine_py.yaml \
            -engine ct2 \
            -out $TMP_OUT_DIR/inference_engine_lm_ct2_outputs  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt
rm $TMP_OUT_DIR/inference_engine_lm_ct2_outputs_file.json
rm $TMP_OUT_DIR/inference_engine_lm_ct2_outputs_list.json

echo -n "  [+] Testing PY SEQ2SEQ inference engine .."
head ${DATA_DIR}/src-test.txt > $TMP_OUT_DIR/src-test.txt
${PYTHON} eole/tests/test_inference_engines.py -model ${TEST_DIR}/test_model \
            -model_type encoder_decoder \
            -input_file $TMP_OUT_DIR/src-test.txt \
            -inference_config_file ${DATA_DIR}/inference-engine_py.yaml \
            -out $TMP_OUT_DIR/inference_engine_seq2seq_py_outputs  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/src-test.txt
rm $TMP_OUT_DIR/inference_engine_seq2seq_py_outputs_file.json
rm $TMP_OUT_DIR/inference_engine_seq2seq_py_outputs_list.json

#
# Tools test
#
echo "[+] Doing tools test..."
# echo -n "  [+] Doing extract vocabulary test..."
# PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH} ${PYTHON} eole/bin/main.py model extract_vocabulary \
#             -model ${TEST_DIR}/test_model -side src \
#             -out_file $TMP_OUT_DIR/vocab.txt >> ${LOG_FILE} 2>&1
# [ "$?" -eq 0 ] || error_exit
# if ! wc -l $TMP_OUT_DIR/vocab.txt | grep -qF  "1002"; then
#     echo -n "wrong word count\n" >> ${LOG_FILE}
#     wc -l $TMP_OUT_DIR/vocab.txt >> ${LOG_FILE}
#     error_exit
# fi
# echo "Succeeded" | tee -a ${LOG_FILE}
# rm $TMP_OUT_DIR/vocab.txt

echo -n "  [+] Doing embeddings to torch test..."
PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH} ${PYTHON} eole/bin/main.py tools embeddings_to_torch \
        -emb_file_enc ${TEST_DIR}/sample_glove.txt \
        -emb_file_dec ${TEST_DIR}/sample_glove.txt \
        -model_path ${TEST_DIR}/test_model \
        -output_file $TMP_OUT_DIR/q_gloveembeddings        >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}
rm $TMP_OUT_DIR/q_gloveembeddings*

echo -n "  [+] Doing extract embeddings test..."
PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH} ${PYTHON} eole/bin/main.py model extract_embeddings \
        -model eole/tests/test_model  >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing architecture rnn Checkpoint Vocabulary Update..."
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.tgt \
            -src_vocab_size 1000 -tgt_vocab_size 1000 \
            -model '{"architecture": "rnn", "hidden_size": 10, "embeddings": {"word_vec_size": 5, "position_encoding_type": None}}' \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 10, "model_path": "'"$TMP_OUT_DIR"'/eole.model", "save_checkpoint_steps": 10}' \
            -report_every 5 \
            >> ${LOG_FILE} 2>&1
sed -i '1s/^/new_tok\t100000000\n/' $TMP_OUT_DIR/eole.vocab.src >> ${LOG_FILE} 2>&1
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.tgt \
            -src_vocab_size 1000 -tgt_vocab_size 1000 \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 20, "train_from": "'"$TMP_OUT_DIR"'/eole.model/step_10", "save_checkpoint_steps": 10, "update_vocab": True, "reset_optim": "states"}' \
            -report_every 5 \
            >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

echo -n "  [+] Testing Checkpoint Vocabulary Update with LM..."
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/lm_data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.src \
            -model '{"layers": 2, "hidden_size": 16, "transformer_ff": 64, "embeddings": {"word_vec_size": 16}, "encoder": None, "decoder": {"decoder_type": "transformer", "heads": 4}}' \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 10, "model_path": "'"$TMP_OUT_DIR"'/lm.eole.model", "save_checkpoint_steps": 10}' \
            -report_every 5 \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            >> ${LOG_FILE} 2>&1
sed -i '1s/^/new_tok2\t100000000\n/' $TMP_OUT_DIR/eole.vocab.src >> ${LOG_FILE} 2>&1
${PYTHON} eole/bin/main.py train \
            -config ${DATA_DIR}/lm_data.yaml \
            -src_vocab $TMP_OUT_DIR/eole.vocab.src \
            -tgt_vocab $TMP_OUT_DIR/eole.vocab.src \
            -model '{"layers": 2, "hidden_size": 16, "transformer_ff": 64, "embeddings": {"word_vec_size": 16}, "encoder": None, "decoder": {"decoder_type": "transformer", "heads": 4}}' \
            -training '{"batch_size": 10, "num_workers": 0, "bucket_size": 1024, "train_steps": 20, "train_from": "'"$TMP_OUT_DIR"'/lm.eole.model/step_10", "save_checkpoint_steps": 10, "update_vocab": True, "reset_optim": "states"}' \
            -report_every 5 \
            -src_vocab_size 1000 \
            -tgt_vocab_size 1000 \
            >> ${LOG_FILE} 2>&1
[ "$?" -eq 0 ] || error_exit
echo "Succeeded" | tee -a ${LOG_FILE}

# Finally, clean up
clean_up

