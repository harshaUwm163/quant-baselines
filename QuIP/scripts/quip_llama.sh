llama_model="llama-3-8b" # "llama-3-8b", "llama-3-70b"
echo num_attn_bits= "$1" num_mlp_bits = "$2"
python  llama.py /data/$llama_model \
                c4 \
                --eval \
                --new-eval \
                --attn_wbits "$1" \
                --mlp_wbits "$2" \
                --quant ldlq \
                --incoh_processing \
                --parent_dir $llama_model \
                --exp_name quip"$1""$2"_act_g128 \
                --seqlen 8192 \
                # --save 
