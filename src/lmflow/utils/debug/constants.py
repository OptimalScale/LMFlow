__gpt2_num_params_lm_head = 50257*768 + 1024*768 # wte, wpe
__gpt2_num_params_gpt2block = 768*2304 + 768*768 + 768*3072 + 768*3072 + 6*768 + 2304 + 3072
__gpt2_num_params_gpt2block_in_pg0 = 768*2304 + 768*768 + 768*3072 + 768*3072 # weight decay (if any)
__gpt2_num_params_gpt2block_in_pg1 = 768 + 768 + 2304 + 768 + 768 + 768 + 3072 + 768 # no weight decay (no matter what) ln_1.weight: 768 ln_1.bias: 768 attn.c_attn.bias: 2304 attn.c_proj.bias: 768 ln_2.weight: 768 ln_2.bias: 768 mlp.c_fc.bias: 3072 mlp.c_proj.bias: 768
__gpt2_param_intervals_pg0 = {"lmhead": (0, __gpt2_num_params_lm_head-1)}
__gpt2_param_intervals_pg0.update({
    f"gpt2block{i}": (
        __gpt2_num_params_lm_head+__gpt2_num_params_gpt2block_in_pg0*i, 
        __gpt2_num_params_lm_head+__gpt2_num_params_gpt2block_in_pg0*(i+1)-1
    ) for i in range(12)
})
__gpt2_param_intervals_pg1 = {
    f"gpt2block{i}": (
        __gpt2_num_params_gpt2block_in_pg1*i, 
        __gpt2_num_params_gpt2block_in_pg1*(i+1)-1
    ) for i in range(12)
}
__gpt2_param_intervals_pg1.update({
    "ln": (__gpt2_num_params_gpt2block_in_pg1*12, __gpt2_num_params_gpt2block_in_pg1*12+768+768-1)
})

GPT2 = {
    "param_name_in_group": [
        {'parameter_names': ['transformer.wte.weight', 'transformer.wpe.weight', 'transformer.h.0.attn.c_attn.weight', 'transformer.h.0.attn.c_proj.weight', 'transformer.h.0.mlp.c_fc.weight', 'transformer.h.0.mlp.c_proj.weight', 'transformer.h.1.attn.c_attn.weight', 'transformer.h.1.attn.c_proj.weight', 'transformer.h.1.mlp.c_fc.weight', 'transformer.h.1.mlp.c_proj.weight', 'transformer.h.2.attn.c_attn.weight', 'transformer.h.2.attn.c_proj.weight', 'transformer.h.2.mlp.c_fc.weight', 'transformer.h.2.mlp.c_proj.weight', 'transformer.h.3.attn.c_attn.weight', 'transformer.h.3.attn.c_proj.weight', 'transformer.h.3.mlp.c_fc.weight', 'transformer.h.3.mlp.c_proj.weight', 'transformer.h.4.attn.c_attn.weight', 'transformer.h.4.attn.c_proj.weight', 'transformer.h.4.mlp.c_fc.weight', 'transformer.h.4.mlp.c_proj.weight', 'transformer.h.5.attn.c_attn.weight', 'transformer.h.5.attn.c_proj.weight', 'transformer.h.5.mlp.c_fc.weight', 'transformer.h.5.mlp.c_proj.weight', 'transformer.h.6.attn.c_attn.weight', 'transformer.h.6.attn.c_proj.weight', 'transformer.h.6.mlp.c_fc.weight', 'transformer.h.6.mlp.c_proj.weight', 'transformer.h.7.attn.c_attn.weight', 'transformer.h.7.attn.c_proj.weight', 'transformer.h.7.mlp.c_fc.weight', 'transformer.h.7.mlp.c_proj.weight', 'transformer.h.8.attn.c_attn.weight', 'transformer.h.8.attn.c_proj.weight', 'transformer.h.8.mlp.c_fc.weight', 'transformer.h.8.mlp.c_proj.weight', 'transformer.h.9.attn.c_attn.weight', 'transformer.h.9.attn.c_proj.weight', 'transformer.h.9.mlp.c_fc.weight', 'transformer.h.9.mlp.c_proj.weight', 'transformer.h.10.attn.c_attn.weight', 'transformer.h.10.attn.c_proj.weight', 'transformer.h.10.mlp.c_fc.weight', 'transformer.h.10.mlp.c_proj.weight', 'transformer.h.11.attn.c_attn.weight', 'transformer.h.11.attn.c_proj.weight', 'transformer.h.11.mlp.c_fc.weight', 'transformer.h.11.mlp.c_proj.weight']}, 
        {'parameter_names': ['transformer.h.0.ln_1.weight', 'transformer.h.0.ln_1.bias', 'transformer.h.0.attn.c_attn.bias', 'transformer.h.0.attn.c_proj.bias', 'transformer.h.0.ln_2.weight', 'transformer.h.0.ln_2.bias', 'transformer.h.0.mlp.c_fc.bias', 'transformer.h.0.mlp.c_proj.bias', 'transformer.h.1.ln_1.weight', 'transformer.h.1.ln_1.bias', 'transformer.h.1.attn.c_attn.bias', 'transformer.h.1.attn.c_proj.bias', 'transformer.h.1.ln_2.weight', 'transformer.h.1.ln_2.bias', 'transformer.h.1.mlp.c_fc.bias', 'transformer.h.1.mlp.c_proj.bias', 'transformer.h.2.ln_1.weight', 'transformer.h.2.ln_1.bias', 'transformer.h.2.attn.c_attn.bias', 'transformer.h.2.attn.c_proj.bias', 'transformer.h.2.ln_2.weight', 'transformer.h.2.ln_2.bias', 'transformer.h.2.mlp.c_fc.bias', 'transformer.h.2.mlp.c_proj.bias', 'transformer.h.3.ln_1.weight', 'transformer.h.3.ln_1.bias', 'transformer.h.3.attn.c_attn.bias', 'transformer.h.3.attn.c_proj.bias', 'transformer.h.3.ln_2.weight', 'transformer.h.3.ln_2.bias', 'transformer.h.3.mlp.c_fc.bias', 'transformer.h.3.mlp.c_proj.bias', 'transformer.h.4.ln_1.weight', 'transformer.h.4.ln_1.bias', 'transformer.h.4.attn.c_attn.bias', 'transformer.h.4.attn.c_proj.bias', 'transformer.h.4.ln_2.weight', 'transformer.h.4.ln_2.bias', 'transformer.h.4.mlp.c_fc.bias', 'transformer.h.4.mlp.c_proj.bias', 'transformer.h.5.ln_1.weight', 'transformer.h.5.ln_1.bias', 'transformer.h.5.attn.c_attn.bias', 'transformer.h.5.attn.c_proj.bias', 'transformer.h.5.ln_2.weight', 'transformer.h.5.ln_2.bias', 'transformer.h.5.mlp.c_fc.bias', 'transformer.h.5.mlp.c_proj.bias', 'transformer.h.6.ln_1.weight', 'transformer.h.6.ln_1.bias', 'transformer.h.6.attn.c_attn.bias', 'transformer.h.6.attn.c_proj.bias', 'transformer.h.6.ln_2.weight', 'transformer.h.6.ln_2.bias', 'transformer.h.6.mlp.c_fc.bias', 'transformer.h.6.mlp.c_proj.bias', 'transformer.h.7.ln_1.weight', 'transformer.h.7.ln_1.bias', 'transformer.h.7.attn.c_attn.bias', 'transformer.h.7.attn.c_proj.bias', 'transformer.h.7.ln_2.weight', 'transformer.h.7.ln_2.bias', 'transformer.h.7.mlp.c_fc.bias', 'transformer.h.7.mlp.c_proj.bias', 'transformer.h.8.ln_1.weight', 'transformer.h.8.ln_1.bias', 'transformer.h.8.attn.c_attn.bias', 'transformer.h.8.attn.c_proj.bias', 'transformer.h.8.ln_2.weight', 'transformer.h.8.ln_2.bias', 'transformer.h.8.mlp.c_fc.bias', 'transformer.h.8.mlp.c_proj.bias', 'transformer.h.9.ln_1.weight', 'transformer.h.9.ln_1.bias', 'transformer.h.9.attn.c_attn.bias', 'transformer.h.9.attn.c_proj.bias', 'transformer.h.9.ln_2.weight', 'transformer.h.9.ln_2.bias', 'transformer.h.9.mlp.c_fc.bias', 'transformer.h.9.mlp.c_proj.bias', 'transformer.h.10.ln_1.weight', 'transformer.h.10.ln_1.bias', 'transformer.h.10.attn.c_attn.bias', 'transformer.h.10.attn.c_proj.bias', 'transformer.h.10.ln_2.weight', 'transformer.h.10.ln_2.bias', 'transformer.h.10.mlp.c_fc.bias', 'transformer.h.10.mlp.c_proj.bias', 'transformer.h.11.ln_1.weight', 'transformer.h.11.ln_1.bias', 'transformer.h.11.attn.c_attn.bias', 'transformer.h.11.attn.c_proj.bias', 'transformer.h.11.ln_2.weight', 'transformer.h.11.ln_2.bias', 'transformer.h.11.mlp.c_fc.bias', 'transformer.h.11.mlp.c_proj.bias', 'transformer.ln_f.weight', 'transformer.ln_f.bias']}
    ],
    "num_params": {
        "lm_head": __gpt2_num_params_lm_head,
        "gpt2block": __gpt2_num_params_gpt2block,
        "gpt2block_in_pg0": __gpt2_num_params_gpt2block_in_pg0,
        "gpt2block_in_pg1": __gpt2_num_params_gpt2block_in_pg1,
    },
    "param_intervals":{
        "pg0": __gpt2_param_intervals_pg0,
        "pg1": __gpt2_param_intervals_pg1
    }
}