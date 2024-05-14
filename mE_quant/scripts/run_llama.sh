llama_model="llama-3-8b" # "llama-3-8b", "llama-3-70b"
python llama_mEquant.py /data/$llama_model \
                c4 \
                --eval \
                --new-eval \
                --parent_dir $llama_model \
                --exp_name mEq_attn_m8e0_mlp_m8e0\
                --save packed_model.pt \
                --seqlen 8192 \
                --ridge_lambda 0.01 \
                --barrier_nu 0.01 \
                --mlp_m 8
