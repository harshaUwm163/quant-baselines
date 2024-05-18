llama_model="llama-3-8b" # "llama-3-8b", "llama-3-70b"
echo num_attn_bits= "$1" num_mlp_bits = "$2"
python llama.py /data/$llama_model \
                c4 \
                --eval \
                --new-eval \
                --attn_wbits "$1" \
                --mlp_wbits "$2" \
                --parent_dir $llama_model \
                --groupsize 128 \
                --exp_name nearest"$1""$2"_act_g128 \
                --seqlen 8192 \
                --nearest


                # --exp_name gptq82_act_gall \
                # --save packed_model.pt \
