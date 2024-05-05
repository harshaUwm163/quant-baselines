python llama.py /data/llama-3-8b \
                c4 \
                --eval \
                --new-eval \
                --wbits 2 \
                --true-sequential \
                --act-order \
                --groupsize 128 \
                --parent_dir llama3-8B \
                --exp_name gptq22_eval \
                --load /data/llama3_felix_results/llama3-8B/gptq22_2024-05-04-22-06-20/packed_model.pt \
                --seqlen 8192 \
