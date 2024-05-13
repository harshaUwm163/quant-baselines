llama_model="hf-llama-3-8b" # "llama-3-8b", "llama-3-70b"
python llama_mEquant.py /data/$llama_model \
                c4 \
                --eval \
                --new-eval \
                --parent_dir $llama_model \
                --exp_name debug_thread \
                --save packed_model.pt \
                --seqlen 8192 \
                --ridge_lambda 0.01 \
                --barrier_nu 0.01 
