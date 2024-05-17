llama_model="llama-3-8b" # "llama-3-8b", "llama-3-70b"
echo num_attn_bits= "$1" num_mlp_bits = "$2"
python llama_mEquant.py /data/$llama_model \
                c4 \
                --eval \
                --new-eval \
                --parent_dir $llama_model \
                --exp_name mEq_attn_m"$1"e0_mlp_m"$2"e0\
                --save packed_model.pt \
                --seqlen 8192 \
                --ridge_lambda 0.01 \
                --barrier_nu 0.01 \
                --attn_m "$1" \
                --mlp_m "$2"
