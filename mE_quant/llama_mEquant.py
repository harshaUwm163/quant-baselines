import argparse
import time
import numpy as np
import torch
import torch.nn as nn

from utils import find_layers, DEV, set_seed, get_loaders, export_quant_table
from me_utils import fp_round, fp_round_error, fi_round, fi_round_error, fp_nmax, fi_max

from texttable import Texttable

from datetime import datetime
import time
import os
import csv
from tqdm import tqdm
import json


def get_llama(model, seqlen=8192):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = seqlen
    return model

def find_scale_mu_fp(  layer, 
                    name,   
                    e_bit, 
                    m_bit,
                    block_size, 
                    support_special_values = False,
                    start_learning_rate = 0.001,
                    s_eps = 1e-5,
                    num_epochs = 100,
                    ridge_lambda = 0.0,
                    barrier_nu = 0.0
                  ):

    def layer_fp_round_error(   params, 
                                x, 
                                e_bit, 
                                m_bit,
                                support_special_values,
                                ridge_lambda,
                                barrier_nu
                                ):
        # depending on the Pytorch broadcasting
        scale = params[0].expand(-1, x.shape[1], -1)
        mu = params[1].expand(-1, x.shape[1], -1)

        return torch.mean(fp_round_error(x, scale, mu, e_bit, m_bit, support_special_values, ridge_lambda, barrier_nu))

    w = layer.weight.data.clone()
    BLOCK = block_size

    b_w = w.reshape((w.shape[0]//BLOCK, BLOCK, w.shape[1]))

    bw_max = torch.max(b_w, axis=1, keepdim=True)[0]
    bw_min = torch.min(b_w, axis=1, keepdim=True)[0]
    
    s = torch.nn.Parameter((bw_max - bw_min) / 2.0 / fp_nmax(e_bit, m_bit, support_special_values))
    mu = torch.nn.Parameter((bw_max + bw_min) / 2.0)
    
    params = [s, mu]
    optim = torch.optim.Adam(params, lr=start_learning_rate)
    
    errs = []
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        loss = layer_fp_round_error(params, b_w, e_bit, m_bit, support_special_values, ridge_lambda, barrier_nu)
        optim.zero_grad()
        loss.backward()
        optim.step()
        params[0].data.clamp_(min=s_eps)
    
        errs.append(loss.item())
        pbar.set_description(f'{name}, loss = {loss.item():.5f}')

    final_s, final_mu = s.detach(), mu.detach()

    del params, optim, s, mu
    torch.cuda.empty_cache()

    return final_s, final_mu

def find_scale_mu_fi(  layer, 
                    name,
                    m_bit,
                    block_size, 
                    start_learning_rate = 0.001,
                    s_eps = 1e-5,
                    num_epochs = 100,
                    ridge_lambda = 0.0,
                    barrier_nu = 0.0
                  ):

    def layer_fi_round_error(   params, 
                                x, 
                                m_bit,
                                ridge_lambda,
                                barrier_nu
                                ):
        scale = params[0].expand(-1, x.shape[1], -1)
        mu = params[1].expand(-1, x.shape[1], -1)

        return torch.mean(fi_round_error(x, scale, mu, m_bit, ridge_lambda, barrier_nu))

    w = layer.weight.data.clone()
    BLOCK = block_size

    b_w = w.reshape((w.shape[0]//BLOCK, BLOCK, w.shape[1]))

    bw_max = torch.max(b_w, axis=1, keepdim=True)[0]
    bw_min = torch.min(b_w, axis=1, keepdim=True)[0]
    
    s = torch.nn.Parameter((bw_max - bw_min) / 2.0 / fi_max(m_bit))
    mu = torch.nn.Parameter((bw_max + bw_min) / 2.0)
    
    params = [s, mu]
    optim = torch.optim.Adam(params, lr=start_learning_rate)
    
    errs = []

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        loss = layer_fi_round_error(params, b_w, m_bit, ridge_lambda, barrier_nu)
        optim.zero_grad()
        loss.backward()
        optim.step()
        params[0].data.clamp_(min=s_eps)
    
        errs.append(loss.item())
        pbar.set_description(f'{name}, loss = {loss.item():.5f}')

    final_s, final_mu = s.detach(), mu.detach()

    del params, optim, s, mu
    torch.cuda.empty_cache()

    return final_s, final_mu



def llama_sequential(model, dev):
    print('Starting ...')

    layers = model.model.layers

    print('Ready.')
    quantizers = {}

    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')

        layer = layers[i].to(dev)
        full = find_layers(layer)
        
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}

            for name in subset:
                if 'attn' in name:
                    e_bit, m_bit = args.attn_e, args.attn_m
                else:
                    e_bit, m_bit = args.mlp_e, args.mlp_m
                    
                # train the scale and mu
                # s, mu = find_scale_mu_fp(  subset[name],
                #                         name,
                #                         e_bit, 
                #                         m_bit,
                #                         args.block_size, 
                #                         support_special_values=args.support_special_values,
                #                         start_learning_rate = args.start_learning_rate,
                #                         s_eps = args.s_eps,
                #                         num_epochs = args.num_epochs,
                #                         ridge_lambda = args.ridge_lambda,
                #                         barrier_nu = args.barrier_nu
                #                     )
                dtype = subset[name].weight.data.type()
                s, mu = find_scale_mu_fi(   subset[name].type(torch.float32), 
                                            name,
                                            m_bit,
                                            args.block_size, 
                                            start_learning_rate = args.start_learning_rate,
                                            s_eps = args.s_eps,
                                            num_epochs = args.num_epochs,
                                            ridge_lambda = args.ridge_lambda,
                                            barrier_nu = args.barrier_nu
                                        )

                # fake quant the layer with the learned scale and mu
                with torch.no_grad():

                    w = subset[name].weight.data.clone().type(torch.float32)
                    BLOCK = args.block_size
                    b_w = w.reshape((w.shape[0]//BLOCK, BLOCK, w.shape[1]))

                    # q_w = fp_round(     b_w,
                    #                     s.expand(-1, b_w.shape[1], -1),
                    #                     mu.expand(-1, b_w.shape[1], -1),
                    #                     e_bit,
                    #                     m_bit,
                    #                     support_special_values=args.support_special_values)
                    q_w = fi_round( b_w, 
                                    s,
                                    mu,
                                    m_bit,
                                    )

                    subset[name].weight.data = q_w.contiguous().view(w.shape[0], w.shape[1]).type(dtype)

                quantizers['model.layers.%d.%s' % (i, name)] = (s.cpu(), mu.cpu())

                del s
                del mu

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

    return ppl.item()


# TODO: perform packing on GPU
def llama_pack(model, quantizers, attn_wbits, mlp_wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, attn_wbits, mlp_wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model


def load_quant(model, checkpoint, attn_wbits, mlp_wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True, seqlen=8192):
    from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, attn_wbits, mlp_wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)

    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = seqlen
    print('Done.')

    return model


def llama_multigpu(model, gpus, gpu_dist):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, 'norm') and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[0])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])

    cache = {'mask': None, 'position_ids': None}

    class MoveModule(nn.Module):

        def __init__(self, module, invalidate_cache):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
            self.invalidate_cache=invalidate_cache

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)

            if cache['mask'] is None or cache['mask'].device != self.dev or self.invalidate_cache:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']

            if cache['position_ids'] is None or cache['position_ids'].device != self.dev or self.invalidate_cache:
                cache['position_ids'] = kwargs['position_ids'].to(self.dev)
            kwargs['position_ids'] = cache['position_ids']
            
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) -1 else gpus[(i-1) // pergpu]), i==0)
    else:
        assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
        assigned_gpus = [0] * (gpu_dist[0]-1)
        for i in range(1, len(gpu_dist)):
            assigned_gpus = assigned_gpus + [i] * gpu_dist[i]

        remaining_assignments = len(layers)-len(assigned_gpus) - 1
        if remaining_assignments > 0:
            assigned_gpus = assigned_gpus + [-1] * remaining_assignments

        assigned_gpus = assigned_gpus + [0]

        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i==0)

    model.gpus = gpus


def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}

    def clear_past(i):

        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i:i + 1], past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if hasattr(model, 'gpus'):
                mem_allocated = sum(torch.cuda.memory_allocated(gpu) for gpu in model.gpus) / 1024 / 1024
            else:
                mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory = max(max_memory, mem_allocated)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
            print('max memory(MiB):', max_memory)


if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--attn_e', type=int, default=0, help='exponent bist for attn')
    parser.add_argument('--attn_m', type=int, default=8, help='mantissa bits for attn')
    parser.add_argument('--mlp_e', type=int, default=0, help='exponent bits for mlp')
    parser.add_argument('--mlp_m', type=int, default=2, help='mantissa bits for mlp')

    parser.add_argument('--block_size', type=int, default=128, help='block size for the quantization')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--test-generation', action='store_true', help='test generation.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')

    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')
    parser.add_argument('--exp_name', type=str, default='debug_thread', help='name of the eperiment')
    parser.add_argument('--parent_dir', type=str, default='llama3-8B', help='parent directory to store the results')
    parser.add_argument('--seqlen', type=int, default=8192, help='context length of the model')

    parser.add_argument('--start_learning_rate', type=float, default=0.001, help='learning rate for learning s, mu')
    parser.add_argument('--s_eps', type=float, default=1e-5, help='smallest scale to prevent it from going to negative')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs for learning s, mu')
    parser.add_argument('--ridge_lambda', type=float, default=0.001, help='ridge lambda for computing quant error')
    parser.add_argument('--barrier_nu', type=float, default=0.001, help='barrier nu for computing quant error')
    parser.add_argument('--support_special_values', action='store_true', help='Whether to support special values during quantization or not')

    args = parser.parse_args()

    results_dir = f'/data/llama3_felix_results/'
    results_dir = os.path.join(results_dir, args.parent_dir)
    args.exp_name = f'{args.exp_name}_'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    exp_dir = os.path.join(results_dir, args.exp_name)

    # print args
    for k,v in args.__dict__.items(): print(k, ':', v)

    DEV = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        # TODO-ahv: needs fixing
        model = load_quant(args.model, args.load, args.attn_wbits, args.mlp_wbits, args.groupsize, seqlen=args.seqlen)
    else:
        model = get_llama(args.model, seqlen=args.seqlen)
        model.eval()

    if not args.load and (args.attn_m < 16 or args.mlp_m < 16):
        tick = time.time()
        quantizers = llama_sequential(model, DEV)
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)

    if args.eval:
        ppls = []
        datasets = ['wikitext2', 'ptb', 'c4']
        if args.new_eval:
            datasets = ['wikitext2', 'ptb-new', 'c4-new']
        for dataset in datasets:
            dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            print(dataset)
            ppl = llama_eval(model, testloader, DEV)
            ppls.append(ppl)

        end_time = time.time()
        elapsed_time = (end_time - start_time)
        results = [args.parent_dir, args.exp_name, args.attn_wbits, args.mlp_wbits, elapsed_time]
        results.extend(ppls)
        os.makedirs(results_dir, exist_ok=True)
        csv_file_path = os.path.join(results_dir, 'results.csv')
        with open(csv_file_path, mode='a', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(results)
    
    if args.test_generation:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        from transformers import LlamaTokenizer, TextStreamer
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
        input_ids = tokenizer(["The capital of New Mexico is"], return_tensors="pt").input_ids.to(gpus[0])
        streamer = TextStreamer(tokenizer)
        with torch.no_grad():
            generated_ids = model.generate(input_ids, streamer=streamer)

    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)

    if args.save:
        os.makedirs(exp_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(exp_dir, 'fakequant_model.pt') )
        torch.save(args, os.path.join(exp_dir, 'args.pt') )
        with open(os.path.join(exp_dir, 'gptq_config.json'), 'w') as h:
            json.dump(args.__dict__, h, indent = 6)

        # save the packed model
        # TODO- packing need to be handled
        # llama_pack(model, quantizers, args.attn_wbits, args.mlp_wbits, args.groupsize)
        torch.save(model.state_dict(), os.path.join(exp_dir, 'packed_model.pt') )

    if not args.observe and args.save_safetensors:
        llama_pack(model, quantizers, args.attn_wbits, args.mlp_wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)
