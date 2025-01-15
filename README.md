# PBRT-MIRROR

## Build

**PBRT** uses **cmake** as its build system. And we add c++17 support based on the original code base. To clone the code base, please run the following command:  

```
git clone --recursive https://github.com/Aaron19960821/pbrt-mirror
```

You can also initialize all submodules when you do not clone the project recursively. Please run the following command:  
```
git submodule update --init --recursive
```

To build this project, please use the following commands:  
```
mkdir build && cd build
cmake ..
make -j
```

Note that **cmake** will not export compile commands defaultly. Thus if you want to set up LSP for this project for better development experience, please add `-DCMAKE_EXPORT_COMPILE_COMMANDS=YES` to your cmake configuration.  

## Major changes

The most significant change should be `lightsampler`, currently we provide the following light samplers:
```
uniform: sample lights uniformly.
SLC: [CY19]stochastic light cut introduced by Cem Yuksel.
NRL: [JP19]naive Q-learning based method.
VARL: periodic Q-learning based method.
BROAS: [PETR18] Bayesian Online Regression for adaptive direct illumination
VABORAS: Variance-Aware variant of BORAS
RIS: Resampled importance sampling.
NRLMIS: MIS on direct illumination with NRL.
VARLMIS: MIS on direct illumination with VARL.
```
To render a scene, one `must` specify a light sampler. We recommend you to use `uniform` if the number of lights is not large and it will provide a slow but rebust result.  
