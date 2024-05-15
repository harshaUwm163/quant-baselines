llama_model="llama-3-8b" # "llama-3-8b", "llama-3-70b"
echo num_mlp_bits = "$1"
python llama.py /data/$llama_model \
                c4 \
                --eval \
                --new-eval \
                --attn_wbits 8 \
                --mlp_wbits "$1" \
                --true-sequential \
                --act-order \
                --parent_dir $llama_model \
                --groupsize 128 \
                --exp_name gptq8"$1"_act_g128 \
                --save packed_model.pt \
                --seqlen 8192


                # --exp_name gptq82_act_gall \
