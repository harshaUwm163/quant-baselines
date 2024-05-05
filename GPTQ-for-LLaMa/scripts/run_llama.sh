llama_model="llama-3-8b" # "llama-3-8b", "llama-3-70b"
python llama.py /data/$llama_model \
                c4 \
                --eval \
                --new-eval \
                --attn_wbits 8 \
                --mlp_wbits 2 \
                --true-sequential \
                --act-order \
                --parent_dir $llama_model \
                --exp_name gptq82_act_gall \
                --save packed_model.pt \
                --seqlen 8192


                # --groupsize 128 \
                # --exp_name gptq82_act_g128 \