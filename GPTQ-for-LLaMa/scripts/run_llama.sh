llama_model="llama-3-8b" # "llama-3-8b", "llama-3-70b"
python llama.py /data/$llama_model \
                c4 \
                --eval \
                --new-eval \
                --attn_wbits 8 \
                --mlp_wbits 2 \
                --true-sequential \
                --act-order \
                --groupsize 128 \
                --parent_dir $llama_model \
                --exp_name gptq22 \
                --save packed_model.pt \
                --seqlen 8192

