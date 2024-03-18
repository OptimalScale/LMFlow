python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 10  0 0 128 # base
python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 10  0 1  128 # Lora
python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 10  0 1  256 # Lora
python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 10  1 0  128 # LISA



python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 512    0 0  128 # base
python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 512    0 1  128 # Lora
python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 512    0 1  256 # Lora
python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 512    1 0  128 # LISA


python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 1024  0 0  128 # base
python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 1024  0 1  128 # Lora
python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 1024  0 1  256 # Lora
python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 1024  1 0  128 # LISA


