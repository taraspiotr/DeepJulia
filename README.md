# DeepJulia
Simple DL package in Julia


## Installation

```Julia
run(`wget 'https://github.com/taraspiotr/DeepJulia/archive/main.zip' -O /tmp/deep_julia.zip`)
run(`unzip -o /tmp/deep_julia.zip -d /usr/local`)
run(`rm /tmp/deep_julia.zip`)
run(`julia -e 'using Pkg; pkg"dev /usr/local/DeepJulia-main; precompile;"'`)

using DeepJulia
```
